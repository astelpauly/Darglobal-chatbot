"""Hybrid retrieval: vector (ChromaDB) + BM25 keyword + RRF merge + light metadata filter.

Returns top-N chunks for a question. Used by the chatbot (app.py).

Usage (test):
  python retrieve.py "Where is the Mulliner located?"
"""
import json, re, sys
from pathlib import Path
from rank_bm25 import BM25Okapi

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CHUNKS_FILE = HERE / "data" / "processed" / "darglobal" / "chunks.jsonl"
DB_DIR = HERE / "data" / "chroma_db"
COLLECTION = "darglobal"
ARABIC_RE = re.compile(r"[؀-ۿ]")
TOKEN_RE = re.compile(r"[\w؀-ۿ]+", re.UNICODE)   # Latin + Arabic word chars


def tokens(text):
    return TOKEN_RE.findall(text.lower())


class Retriever:
    """Loads chunks once, keeps BM25 index and ChromaDB handle in memory."""

    def __init__(self, embedder):
        self.embedder = embedder                 # already-loaded BGE-M3 (shared with app.py)
        self.chunks = [json.loads(l) for l in CHUNKS_FILE.read_text(encoding="utf-8").split("\n") if l.strip()]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokens(c["text"]) for c in self.chunks])

        import chromadb
        self.col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection(COLLECTION)
        print(f"retriever ready: {len(self.chunks)} chunks, {self.col.count()} embeddings")

    # ---------- the two searches
    def _embed_cached(self, question):
        """Cache question embeddings so repeat/similar queries are instant."""
        if not hasattr(self, "_cache"):
            self._cache = {}
        if question not in self._cache:
            if len(self._cache) > 500:            # bounded LRU-ish
                self._cache.pop(next(iter(self._cache)))
            self._cache[question] = self.embedder.encode([question]).tolist()
        return self._cache[question]

    def vector_search(self, question, n, where=None):
        qv = self._embed_cached(question)
        r = self.col.query(query_embeddings=qv, n_results=n, where=where or None,
                           include=["metadatas", "distances"])
        return [(mid, 1 - dist)                   # cosine distance -> similarity
                for mid, dist in zip(r["ids"][0], r["distances"][0])]

    def bm25_search(self, question, n, language=None):
        scores = self.bm25.get_scores(tokens(question))
        order = scores.argsort()[::-1]
        out = []
        for i in order:
            if scores[i] <= 0:
                break
            c = self.chunks[i]
            if language and c.get("language") != language:
                continue
            out.append((c["id"], float(scores[i])))
            if len(out) >= n:
                break
        return out

    # ---------- merge
    @staticmethod
    def rrf(result_lists, k=60):
        """Reciprocal Rank Fusion: ignore raw scores, use only rank positions."""
        score = {}
        for results in result_lists:
            for rank, (cid, _) in enumerate(results, start=1):
                score[cid] = score.get(cid, 0) + 1 / (k + rank)
        return sorted(score.items(), key=lambda x: x[1], reverse=True)

    # ---------- main call
    def retrieve(self, question, top_k=20):
        """Returns up to top_k chunks (dicts) ranked best first."""
        # light filter: only language, and only when the question is clearly Arabic
        lang = "ar" if ARABIC_RE.search(question) else None
        where = {"language": lang} if lang else None

        vec = self.vector_search(question, n=top_k, where=where)
        bm = self.bm25_search(question, n=top_k, language=lang)

        # fallback: if filter returned nothing, retry without it
        if lang and not vec and not bm:
            vec = self.vector_search(question, n=top_k)
            bm = self.bm25_search(question, n=top_k)

        merged = self.rrf([vec, bm])

        # apply chunk weight (blogs = 0.6, projects = 1.0) so project pages win on ties
        out = []
        for cid, s in merged:
            c = self.by_id.get(cid)
            if not c:
                continue
            out.append({**c, "retrieval_score": s * c.get("weight", 1.0)})
        out.sort(key=lambda c: c["retrieval_score"], reverse=True)
        return out[:top_k]


# --------------------------------------------------- CLI test
if __name__ == "__main__":
    from embed import BgeM3
    r = Retriever(BgeM3())

    def show(q, n=20):
        """Shows all 20 candidates (what the reranker would then pick 5 from)."""
        print(f"\nQ: {q}")
        print("-" * 70)
        for i, c in enumerate(r.retrieve(q, top_k=n), 1):
            snippet = c["text"].replace("\n", " ")
            print(f"[{i:2}] score={c['retrieval_score']:.4f}  {c['chunk_type']:13} "
                  f"lang={c['language']}")
            print(f"     {snippet[:180]}...")
            print(f"     -> {c['url']}\n")

    if len(sys.argv) > 1:                            # one-shot: python retrieve.py "query"
        show(" ".join(sys.argv[1:]))
    else:                                            # interactive loop
        print("\nType a question (shows top-20 candidates). "
              "Blank line or Ctrl+C to quit.\n")
        while True:
            try:
                q = input("Q> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not q:
                break
            show(q)
