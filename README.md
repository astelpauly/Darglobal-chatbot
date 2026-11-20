# DarGlobal Property Assistant

A retrieval-augmented chatbot that answers questions about Dar Global's luxury
real estate projects, partners, and news, in English and Arabic. Built from
public content scraped from darglobal.co.uk.

Live: https://darglobal-chatbot-391851064699.us-central1.run.app

## Pipeline

1. **Scrape** – 611 pages crawled with Crawl4AI (`scripts/scrape.py`).
2. **Chunk** – heading-aware chunks (~1,000 chars) with name tags, project
   cards, and per-FAQ chunks: 4,768 chunks (`scripts/chunk.py`).
3. **Embed** – BGE-M3 (ONNX, multilingual, 1024-dim) → ChromaDB with HNSW
   cosine index (`embed.py`).
4. **Retrieve** – hybrid: vector + BM25 + Reciprocal Rank Fusion + language
   filter + page-type weight (`retrieve.py`).
5. **Rerank** – Jina Reranker v2 multilingual (ONNX), dynamic top-N
   (`rerank.py`).
6. **Generate** – Groq LLM chain (gpt-oss-120b → qwen3-32b → gpt-oss-20b) with
   grounded prompt + source citations (`app.py`).
7. **Observability** – LangSmith traces for every step.

## Project layout

```
.
├── app/                # runtime (Docker COPY target)
│   ├── main.py         # FastAPI backend
│   ├── embed.py        # BGE-M3 ONNX wrapper
│   ├── retrieve.py     # vector + BM25 + RRF
│   ├── rerank.py       # Jina reranker + dynamic top-N
│   └── index.html      # chat UI
├── requirements.txt
├── Dockerfile          # Cloud Run image
├── scripts/            # one-off pipeline (not in prod image)
│   ├── scrape.py
│   └── chunk.py
├── evals/              # offline evaluation
│   ├── evals.jsonl
│   └── run_evals.py
├── docs/
│   └── TECHNICAL_WRITEUP.md
└── data/
    ├── raw/            # crawl output (gitignored)
    ├── processed/
    │   └── darglobal/chunks.jsonl
    └── chroma_db/      # pre-built ChromaDB 1.x index
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate         # Windows
pip install -r requirements.txt
# put GROQ_API_KEY=... in .env
python -m uvicorn app.main:app --port 7860
```

## Build & run Docker

```bash
docker build -t darglobal .
docker run -p 8080:8080 --env-file .env darglobal
```

## Deploy

Pushes to `main` on GitHub trigger Cloud Build → Cloud Run (us-central1).
Cloud Run keeps every revision, so a failed build leaves traffic on the last
healthy one. Manual rollback: Cloud Run console → Revisions → *Manage traffic*.

## Env vars

- `GROQ_API_KEY` – required
- `LANGSMITH_API_KEY` – optional (tracing)
- `LANGSMITH_TRACING=true` – optional
- `LANGSMITH_PROJECT=darglobal-chatbot` – optional

## Evaluation

```bash
python evals\run_evals.py http://localhost:7860
```
