"""Run the evaluation set against a live chatbot URL (local or Hugging Face).

Measures:
  - keyword_hit: does the answer contain the must_contain words?
  - source_hit:  did the retrieved sources include the expected page?
  - latency_s:   time per answer
  - faithfulness: judged by Groq (is every claim supported by the sources?)
  - relevance:    judged by Groq (does the answer address the question?)

Uploads each run to LangSmith as a trace (if LANGSMITH_API_KEY is set).

Usage:
  python run_evals.py                           # tests http://localhost:7860
  python run_evals.py https://<name>.hf.space   # tests the deployed Space
"""
import asyncio, json, os, sys, time
from pathlib import Path
import httpx

try:
    from dotenv import load_dotenv; load_dotenv()
except ImportError:
    pass

HERE = Path(__file__).resolve().parent
BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:7860").rstrip("/")
EVAL_FILE = HERE / "evals.jsonl"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT = """You are grading a chatbot answer against the provided context.
Return JSON only: {"faithfulness": 0-5, "relevance": 0-5, "note": "<brief>"}
- faithfulness: 5 = every claim is supported by context, 0 = made up
- relevance: 5 = fully answers the question, 0 = off-topic
QUESTION: {q}
CONTEXT: {c}
ANSWER: {a}"""


async def ask(client, question):
    t = time.time()
    r = await client.post(f"{BASE}/api/chat", json={"message": question, "history": []}, timeout=60)
    r.raise_for_status()
    j = r.json()
    return j, time.time() - t


async def judge(client, q, answer, sources):
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        return {"faithfulness": None, "relevance": None, "note": "no GROQ_API_KEY"}
    ctx = "\n".join(f"[{s['n']}] {s.get('title','')} - {s['url']}" for s in sources)
    prompt = JUDGE_PROMPT.format(q=q, c=ctx[:2000], a=answer[:2000])
    try:
        r = await client.post(GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={"model": JUDGE_MODEL, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=30)
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        return {"faithfulness": None, "relevance": None, "note": str(e)}


async def main():
    cases = [json.loads(l) for l in EVAL_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"running {len(cases)} evals against {BASE}")
    results = []
    async with httpx.AsyncClient() as client:
        for i, case in enumerate(cases, 1):
            try:
                resp, latency = await ask(client, case["question"])
            except Exception as e:
                print(f"[{i}] ERROR: {e}")
                results.append({**case, "error": str(e)})
                continue
            answer = resp.get("answer", "")
            sources = resp.get("sources", [])
            must = case.get("must_contain", [])
            keyword_hit = all(k.lower() in answer.lower() for k in must) if must else None
            source_hit = (case.get("page_url") in [s["url"] for s in sources]) if case.get("page_url") else None
            grade = await judge(client, case["question"], answer, sources)
            r = {**case, "answer": answer[:300], "latency_s": round(latency, 2),
                 "keyword_hit": keyword_hit, "source_hit": source_hit, **grade,
                 "model": resp.get("model")}
            results.append(r)
            print(f"[{i:2}] {latency:4.1f}s  kw={keyword_hit}  src={source_hit}  "
                  f"faith={grade.get('faithfulness')}  rel={grade.get('relevance')}  "
                  f"| {case['question'][:60]}")

    # summary
    scored = [r for r in results if r.get("faithfulness") is not None]
    kw = [r for r in results if r.get("keyword_hit") is not None]
    out = HERE / "eval_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n===== SUMMARY =====")
    print(f"keyword hit:    {sum(r['keyword_hit'] for r in kw)}/{len(kw)}")
    print(f"avg faithfulness: {sum(r['faithfulness'] for r in scored)/len(scored):.2f}/5")
    print(f"avg relevance:    {sum(r['relevance'] for r in scored)/len(scored):.2f}/5")
    print(f"avg latency:      {sum(r['latency_s'] for r in results)/len(results):.2f}s")
    print(f"saved -> {out}")

if __name__ == "__main__":
    asyncio.run(main())
