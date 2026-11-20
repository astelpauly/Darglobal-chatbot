# Dar Global Property Assistant — Technical Write-up

**Live app:** https://darglobal-chatbot-391851064699.us-central1.run.app
**Repository:** https://github.com/astelpauly/Darglobal-chatbot
**Author:** Astel Pauly

---

## 1. Overview

A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers
questions about Dar Global's luxury real estate projects, partners, and news,
in **English and Arabic**. Built end-to-end from publicly available data on
darglobal.co.uk.

**Pipeline:**

```
scrape  →  chunk  →  embed  →  retrieve (hybrid)  →  rerank  →  LLM  →  grounded answer
```

**Stack:** Python 3.11, Crawl4AI, BGE-M3 (ONNX), Jina Reranker v2 (ONNX),
ChromaDB 1.x, rank-bm25, FastAPI, Groq LLM, LangSmith, Docker, Google Cloud Run.

---

## 2. Data Collection — Scraping

### Tool
**Crawl4AI 0.9.5** (headless Chromium via Playwright). Chosen over plain
BeautifulSoup for:
- JavaScript rendering (Dar Global's pages are client-side hydrated)
- Built-in markdown conversion
- Content-aware pruning (noise removal)

### Strategy
- **Sitemap-driven**: start from `/sitemap.xml`, follow every real page
- **Breadth-first link discovery** from header, menu, footer, and in-page `<a href>`
- **Polite crawling**: random pauses between 5–12 s, exponential backoff on 429/403
- **Imperva/Incapsula detection** handled: blocked pages requeued, skipped after 2 failures
- **Resumable**: `pages.jsonl` + `redirects.txt` checkpointed on every save

### Cleaning (in `scrape.py`)
Each page is:
1. Fetched as HTML + rendered markdown
2. Stripped of tracker images, share buttons, newsletter CTAs, dropdown filter lines, duplicate cards
3. Section-level cuts: *"Register your interest"*, *"Live All In"* footer marketing block
4. Key facts parsed from `<ul>` blocks into structured `key_facts` metadata
5. Card patterns (project tiles, article thumbnails, partner logos) normalised

### Result
- **611 pages** scraped (467 of 468 sitemap pages — 1 blocked)
- **39 empty/broken pages** identified and excluded downstream
- **79 Arabic pages**, 268 blog posts, 60 project pages, plus partners/press/insights

**Output:** `data/processed/darglobal/pages.jsonl`

---

## 3. Chunking Strategy

### Approach: heading-aware recursive chunking

Flat character-based chunking breaks on headings and mixes topics. Semantic
chunking is slow and unstable. Website pages already have strong structure, so
I used it.

### Rules (in `chunk.py`)
1. **Clean per-page**: shorten link-card blobs, remove image URLs, drop pages under 30 words
2. **Split on Markdown headings** (`#`, `##`, `###`) — one section per topic
3. **Size-adjust:**
   - Target ~**1,000 characters** per chunk (~250 tokens)
   - Smaller than 200 chars → join with next (avoid useless 1-line chunks)
   - Larger than target → recursive split at paragraph → sentence → word
   - **150 character overlap** on splits so a fact crossing a boundary is preserved
4. **Name-tag every chunk**: `Project Name › Section Name` prepended to the text
   - Context is preserved for search and the LLM
5. **Special chunk types:**
   - `project_card`: one per project, built from `key_facts` + location + status + related projects
   - `faq`: each question + answer kept as one indivisible chunk
   - `table`: kept whole; split only if >2,500 chars (repeating header row)
6. **Drop remaining micro-junk** (chunks under 120 chars after name-tag)

### Metadata per chunk
`url, title, section, page_type, language, location, chunk_type, chunk_index, weight`

### `weight` metadata (page-type bias)
To stop 268 blog posts from drowning out 60 project pages:
| page_type | weight |
|---|---|
| project, faq, about, contact | 1.0 |
| product, partner, home, division | 0.9 |
| project_list, insight | 0.7–0.8 |
| **blog, press** | **0.6** |

Applied as a multiplier on retrieval scores — blogs still win on genuinely
news-shaped questions.

### Result
- **4,768 chunks** (72 project cards, 635 FAQ chunks, 9 tables, 4,052 text)
- **4,384 English · 358 Arabic · 26 mixed**
- Median size **~597 chars**, max 1,371 chars

**Output:** `data/processed/darglobal/chunks.jsonl`

---

## 4. Embeddings

### Model: **BGE-M3** (BAAI, 1024-dim, multilingual, ONNX)

Chosen because:
- **Multilingual (100+ languages)** — Arabic ↔ English queries share one vector space
- 1,024 dim dense embeddings, strong quality on both languages
- 8K-token context (we use 512, enough for our ~250-token chunks)
- Open and free; runs locally

### ONNX, not PyTorch
Windows Smart App Control blocked PyTorch binaries on the dev machine. Switched
to **ONNX Runtime** (Microsoft-signed, allowed) using the official BGE-M3 ONNX
export from BAAI. Same vectors, no quality loss.

### Storage: **ChromaDB 1.x** with **HNSW** cosine index
- 50 MB total (`chroma.sqlite3` + `data_level0.bin` + HNSW link lists)
- Indexed for sub-50 ms vector search

### Query cache
- LRU cache (max 500) for question embeddings — repeat queries skip the model entirely (0 ms vs ~500 ms)

**Output:** `data/chroma_db/`

---

## 5. Retrieval — Hybrid

Vector search alone misses exact names and numbers (e.g. "Mulliner", "AED 2.5M").
Keyword search alone misses paraphrases ("handover" vs "when can I move in?").
Combined, they cover each other.

### Pipeline (in `retrieve.py`)

**1. Light metadata filter**
- Only one rule: **Arabic script detected in question → `language = "ar"`**
- Chosen over LLM-based query understanding for speed, cost, and determinism
- Falls back to no filter if filtered results are empty

**2. Two searches in parallel**
- **Dense (vector)**: BGE-M3 encode → ChromaDB HNSW cosine top-20
- **Sparse (BM25)**: tokenised Latin + Arabic word chars, `rank-bm25` on all chunks, top-20

**3. Merge with Reciprocal Rank Fusion**
```
score(chunk) = Σ 1 / (60 + rank_in_list)
```
RRF uses only rank, not raw scores, so vector and BM25 scores (different
scales) are combined fairly. The `k=60` smoothing flattens top ranks so
agreement between both searches outweighs one list's #1.

**4. Weight multiplier**
Final score = `rrf_score × chunk.weight` — nudges project pages above blogs
when scores are close.

---

## 6. Reranking

Top-20 from retrieval is sent to a **cross-encoder** that scores each chunk
against the question *together* (not independently). Much more accurate than
embedding similarity.

### Model: **jinaai/jina-reranker-v2-base-multilingual** (ONNX)
Chosen over BGE-reranker-v2-m3 because:
- Multilingual (Arabic + English) — essential
- **Official ONNX export** (BGE-reranker had none)
- 550 MB vs 2.3 GB (faster download, lighter image)

### Dynamic top-N selection (in `rerank.py`)
A fixed top-5 is wrong — some questions deserve 3 chunks, list questions
deserve 10. Rules:

1. Score every candidate with the cross-encoder (logits)
2. Convert to probabilities via sigmoid (easier to threshold)
3. Keep chunks that pass **both**:
   - Absolute: `prob >= 0.4`
   - Relative: `prob >= 0.5 × top_prob`
4. Clamp to **[min 3, max 10]** — never flood the LLM context

This means "Where is Mulliner?" gets ~3 chunks; "List all Oman projects" gets
8–10 — automatically, based on reranker agreement.

### Graceful fallback
`try_load()` catches ONNX / tokenizer errors — if the reranker fails, the app
runs on top-5 from RRF alone. Chatbot never breaks.

---

## 7. Generation — LLM

### Primary: **Groq**, all-free chain
```
1. openai/gpt-oss-120b   ← smart, long answers
2. qwen/qwen3-32b        ← strong Arabic
3. openai/gpt-oss-20b    ← fast fallback
```

Automatic fallback: if a model hits a 429 (rate limit) or 5xx, the app
transparently moves to the next. User sees no error.

### Grounded prompting
```
Rules:
- Use only the CONTEXT. Do NOT invent prices, dates, numbers, or facts.
- Cite sources at the end as "Sources: [1], [2]" using the numbered list.
- Reply in the user's language (English or Arabic).
```

### Citations
Chunks are sent to the LLM as `[1] ...`, `[2] ...`. The **URLs never go to
the model** (prevents hallucinated links; saves tokens). The backend attaches
them to the response, and the frontend renders them as clickable gold links.

### Conversation memory
Last 6 turns are kept and prepended to each request (bounded to prevent context
bloat).

### Output post-processing
- Strip internal span markers (`【1†L1-L3】`) that gpt-oss sometimes emits
- Remove empty "Sources:" lines produced by that strip
- Pass through our Markdown renderer safely (DOMPurify)

---

## 8. Frontend

### Design
Custom HTML, CSS, JavaScript — intentionally **not Gradio or Streamlit**. The
chatbot is for a luxury real-estate brand; the UI must reflect that.

- **Dar Global brand**: black (#0b0b0c) + champagne gold (#c9a96a)
- **Cormorant Garamond** serif for the brand and tagline; **Inter** for body
- Welcome panel with 4 suggested questions (English + Arabic sample)
- Gold up-arrow send button, animated typing dots
- **Arabic RTL support** auto-detected from the user's message
- Markdown rendering via `marked.js`, sanitised with `DOMPurify`
- Numbered gold source links below each answer
- Mobile responsive

### Robust error handling
- Gateway timeouts return plaintext, not JSON — the frontend detects this and
  shows *"The server is still starting up (first visit can take ~1 min)"*
  instead of a JSON parse crash.

---

## 9. Backend — API

**FastAPI** (`app.py`), async throughout.

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET  | `/`          | Chat page (reads `index.html` each request so edits apply without restart) |
| POST | `/api/chat`  | `{message, history}` → `{answer, sources, model}` |
| GET  | `/api/health`| `{ready, loading, error, chunks, embeddings, reranker}` |

### Lazy model loading
Models load in a **background thread** so port 8080 opens in <1 second. The
Cloud Run health probe passes instantly; the first visitor's chat request
waits up to 5 min for models to be ready (`_wait_ready(300)`).

Without this, Cloud Run killed the container with
*"failed to listen on port 8080 within the allocated timeout"*.

---

## 10. Containerisation

**Dockerfile** on `python:3.11-slim`.

Key decisions:
- `HF_HOME=/app/hf_cache` — **not `/tmp`** (Cloud Run mounts `/tmp` as tmpfs,
  wiping any build-time downloads)
- `PYTHONUNBUFFERED=1` — logs flush immediately for Cloud Run log viewer
- `ORT_THREADS=2` — ONNX Runtime pinned to container CPU count
- Pre-download BGE-M3 + Jina reranker at build time → first user request has
  no HF download wait
- `.dockerignore` keeps `.env`, `venv/`, raw scrape outputs out of the image

**Image size:** ~3.5 GB (most of that is the two models — the trade-off for
sub-minute cold starts).

---

## 11. Deployment — Google Cloud Run

**Why Cloud Run, not Hugging Face Spaces?** HF Spaces now charges for the
Docker SDK. Cloud Run keeps Docker **free within the monthly quota**.

### Continuous deployment from GitHub
- Code lives at https://github.com/astelpauly/Darglobal-chatbot
- Cloud Build trigger set on `main` branch
- Every `git push` → build → push to Artifact Registry → deploy new revision
- Same URL across deploys

### Service config
| Setting | Value | Reason |
|---|---|---|
| Memory | 8 GiB | BGE-M3 (3 GB) + Jina (1.5 GB) + ChromaDB + runtime ≈ 6 GB |
| CPU | 4 vCPU | Required for 8 GB tier; also helps ONNX multi-thread |
| Min instances | 0 | **$0 when idle** |
| Max instances | 2 | Cost ceiling |
| Billing | **Instance-based** | Request-based throttles the background model-loading thread — models never finish loading |
| Request timeout | 900 s | Safety for slow cold starts |
| Concurrency | 4 | Our RAM budget safely handles 4 parallel questions |
| Startup CPU boost | on | Faster first-request latency |

### Secrets
`GROQ_API_KEY` and `LANGSMITH_API_KEY` stored in **Google Secret Manager**,
referenced into the container as env vars with least-privilege IAM.

---

## 12. Observability — LangSmith

Every request is traced:

```
chat (chain)  ───────────────────  ~5 s
  ├─ retrieve (retriever)  ─────  ~0.8 s   (shows the top-10 chunks)
  └─ groq_call (llm)  ──────────  ~1.5 s   (model, tokens, cost)
```

Enables us to see for every query:
- Which chunks were picked
- Which Groq model answered
- Time per step
- Any failures

`@traceable` decorators were added without changing business logic — a no-op
when `LANGSMITH_API_KEY` is absent (so local dev stays zero-config).

---

## 13. Evaluation

### Test set: `evals.jsonl` — 30 questions
Covers:
- Single-fact lookup ("Where is the Mulliner?")
- Multi-source lists ("Projects in Oman")
- Numeric facts ("Area in square metres?")
- Price / availability ("Is Mulliner still available?")
- Partner relationships ("Who designed Azure interiors?")
- Arabic variants (4 questions)
- Impossible questions ("Weather in London?") — tests grounded refusal

### Metrics (`run_evals.py`)
| Metric | How |
|---|---|
| **keyword_hit** | Does the answer contain required facts? |
| **source_hit**  | Did retrieval return the expected page URL? |
| **faithfulness (0–5)** | Groq `openai/gpt-oss-120b` as LLM-judge: every claim supported by context? |
| **relevance (0–5)** | LLM-judge: does the answer address the question? |
| **latency**   | Wall-clock per answer |

Results saved to `eval_results.json` and traced in LangSmith for comparison
across pipeline changes.

---

## 14. Known trade-offs

| Area | Decision | Trade-off |
|---|---|---|
| Local-first embedding | Re-embed on laptop (~80 min) instead of Cloud Build (~2 h at $) | One-time local pain, free deployment |
| Image size 3.5 GB | Pre-bake models | Faster cold start, ~$0.30/month Artifact Registry storage |
| Instance-based billing | Needed for background model loading | Slightly higher per-session cost (still $0 idle) |
| Light metadata filter (regex) | No LLM-based query understanding | Saves one LLM call per question; covers 90% of cases |
| Dynamic top-N on reranker | Rather than fixed 5 | Simpler cases save tokens; list cases get more context |

---

## 15. What's next

1. **Wasalt integration** — same pipeline, larger-scale crawl (needs direction on priority sections)
2. **Streaming responses** — stream Groq tokens via SSE for perceived speed
3. **Query understanding LLM step** — for ambiguous questions where the simple
   language filter isn't enough
4. **Quantized INT8 models** — 2–3× smaller + faster, small quality drop
5. **Analytics dashboard** — surface LangSmith data (top queries, failure types) in-app

---

## 16. Reproducibility

```bash
git clone https://github.com/astelpauly/Darglobal-chatbot
cd Darglobal-chatbot
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
# add .env with GROQ_API_KEY, LANGSMITH_API_KEY
python app.py                                 # http://localhost:7860
```

To redeploy: `git push` → Cloud Build + Cloud Run handle the rest.

---

*Thank you for reviewing this submission. I'm happy to walk through any part
of the pipeline on a call.*
