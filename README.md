
# Dar Global Property Assistant

A retrieval-augmented chatbot that answers questions about Dar Global's luxury
real estate projects, partners, and news, in English and Arabic. Built from
publicly available content scraped from darglobal.co.uk.

## Pipeline
1. **Scrape** – 611 pages crawled with Crawl4AI.
2. **Chunk** – heading-aware chunks (~1,000 chars) with name tags, project
   cards, and per-FAQ chunks (4,768 chunks total).
3. **Embed** – BGE-M3 (ONNX, multilingual) → ChromaDB with HNSW cosine index.
4. **Retrieve** – hybrid: vector + BM25 + Reciprocal Rank Fusion + language
   filter.
5. **Rerank** – Jina Reranker v2 multilingual (ONNX), dynamic top-N.
6. **Generate** – Groq (openai/gpt-oss-120b, qwen3-32b, gpt-oss-20b), with
   grounded prompt and source citations.
7. **Observability** – LangSmith traces for every step.

## Env vars (set as Space secrets)
- `GROQ_API_KEY` – required
- `LANGSMITH_API_KEY` – optional
- `LANGSMITH_TRACING=true` – optional
- `LANGSMITH_PROJECT=darglobal-chatbot` – optional
