"""DarGlobal RAG chatbot.

FastAPI backend + custom HTML/JS chat page.
Pipeline: retrieve (vector + BM25 + RRF) -> rerank -> LLM answer -> cite sources.

Endpoints:
  GET  /             chat page (index.html)
  POST /api/chat     body = {"message": "...", "history": [...]}  -> {"answer", "sources"}
  GET  /api/health   {"status": "ok", ...}

LLM: Groq, with fallback chain.
Deploy: Docker -> Google Cloud Run (port set by $PORT env var).
"""
import os, re, json
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from dotenv import load_dotenv; load_dotenv()      # read .env (local only)
except ImportError:
    pass

import httpx
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from embed import BgeM3
from retrieve import Retriever, ARABIC_RE
from rerank import try_load

try:
    from langsmith import traceable
except ImportError:
    def traceable(*a, **kw):
        if a and callable(a[0]): return a[0]
        return lambda f: f

HERE = Path(__file__).resolve().parent
TOP_K = int(os.getenv("TOP_K", "20"))
TOP_N = int(os.getenv("TOP_N", "5"))
TIMEOUT = 30

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_CHAIN = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "openai/gpt-oss-20b",
]

SYSTEM_PROMPT = """You are DarGlobal's property assistant. You answer only from the
provided CONTEXT, which contains excerpts from darglobal.co.uk.

Rules:
- Use only the CONTEXT. Do NOT invent prices, dates, numbers, or facts.
- If the CONTEXT does not have the answer, say so briefly and suggest what to search instead.
- Cite sources at the end as "Sources: [1], [2]" using the numbered list in the CONTEXT.
- Keep answers short and factual. Use bullet points for lists.
- Reply in the user's language (English or Arabic)."""

state = {"ready": False, "loading": False, "error": None}


def _load_models_blocking():
    """Load BGE-M3 + retriever + reranker. Runs in a background thread so the
    HTTP port opens immediately and the Cloud Run health check passes."""
    try:
        state["loading"] = True
        print("loading models (background)...", flush=True)
        state["embedder"] = BgeM3()
        state["retriever"] = Retriever(state["embedder"])
        state["reranker"] = try_load()
        state["http"] = httpx.AsyncClient(timeout=TIMEOUT)
        state["ready"] = True
        state["loading"] = False
        print("ready", flush=True)
    except Exception as e:
        state["error"] = str(e)
        state["loading"] = False
        print(f"startup failed: {e}", flush=True)


@asynccontextmanager
async def lifespan(app):
    import threading
    threading.Thread(target=_load_models_blocking, daemon=True).start()
    yield
    http = state.get("http")
    if http is not None:
        await http.aclose()


app = FastAPI(title="DarGlobal Chatbot", lifespan=lifespan)


async def _wait_ready(max_wait=300):
    """Called by endpoints that need the models. Waits for background load."""
    import asyncio
    for _ in range(max_wait * 2):
        if state["ready"]:
            return True
        if state["error"]:
            raise RuntimeError(f"startup failed: {state['error']}")
        await asyncio.sleep(0.5)
    raise RuntimeError("models still loading, try again in a moment")


class ChatIn(BaseModel):
    message: str
    history: list[dict] = []


def build_context(chunks):
    parts, sources = [], []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] {c['text']}")
        sources.append({"n": i, "title": c.get("title"), "url": c["url"],
                        "section": c.get("section", "")})
    return "\n\n".join(parts), sources


@traceable(run_type="llm", name="groq_call")
async def call_llm(messages):
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = ""
    for model in LLM_CHAIN:
        try:
            r = await state["http"].post(GROQ_URL, headers=headers,
                                         json={"model": model, "messages": messages,
                                               "temperature": 0.2, "max_tokens": 600})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"], model
            last_err = f"{model} -> {r.status_code}: {r.text[:200]}"
            print(last_err)
        except Exception as e:
            last_err = f"{model} -> {e}"
            print(last_err)
    raise RuntimeError(f"all Groq models failed; last: {last_err}")


@app.get("/api/health")
async def health():
    r = state.get("retriever")
    return {"status": "ok",
            "ready": state.get("ready", False),
            "loading": state.get("loading", False),
            "error": state.get("error"),
            "chunks": len(r.chunks) if r else 0,
            "embeddings": r.col.count() if r else 0,
            "reranker": state.get("reranker") is not None}


@traceable(run_type="retriever", name="retrieve")
def do_retrieve(q):
    chunks = state["retriever"].retrieve(q, top_k=TOP_K)
    if state["reranker"]:
        chunks = state["reranker"].rerank(q, chunks)
    else:
        chunks = chunks[:TOP_N]
    return chunks


@app.post("/api/chat")
@traceable(run_type="chain", name="chat")
async def chat(inp: ChatIn):
    q = inp.message.strip()
    if not q:
        return JSONResponse({"error": "empty message"}, 400)

    try:
        await _wait_ready()
    except Exception as e:
        return JSONResponse({"error": str(e)}, 503)

    chunks = do_retrieve(q)
    context, sources = build_context(chunks)
    history = [m for m in (inp.history or []) if m.get("role") in ("user", "assistant")][-6:]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user",
                     "content": f"CONTEXT:\n{context}\n\nQUESTION: {q}"})

    try:
        answer, used = await call_llm(messages)
    except Exception as e:
        return JSONResponse({"error": str(e)}, 502)
    answer = re.sub(r"【[^】]*】", "", answer)
    answer = re.sub(r"(?im)^\s*\**sources?\**[\s:]*[,\s]*$\n?", "", answer).strip()
    return {"answer": answer, "sources": sources, "model": used}


@app.get("/", response_class=HTMLResponse)
async def root():
    p = HERE / "index.html"
    return p.read_text(encoding="utf-8") if p.exists() else "<h1>chat page missing</h1>"


if __name__ == "__main__":
    import uvicorn
    # Cloud Run sets PORT (usually 8080); local default 7860
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")))
