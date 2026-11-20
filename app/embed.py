"""Turn every chunk into numbers (embedding) with BGE-M3 and store them in ChromaDB.

Input : data/processed/darglobal/chunks.jsonl
Output: data/chroma_db/   (collection "darglobal", cosine HNSW index)

Usage:
  python embed.py --preview                 # embed ONE chunk and show the result, writes nothing
  python embed.py --preview --pick fendi    # preview the first chunk whose url contains "fendi"
  python embed.py                           # embed everything (resumes if stopped)
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

HERE = Path(__file__).resolve().parent
CHUNKS = HERE / "data" / "processed" / "darglobal" / "chunks.jsonl"
DB_DIR = HERE / "data" / "chroma_db"
COLLECTION = "darglobal"
MODEL = "BAAI/bge-m3"
MAX_TOKENS = 512      # our chunks are ~1,000 chars (~250 tokens); a lower limit saves memory
BATCH = 8             # small batches -> low memory use on an 8 GB laptop


def load_chunks():
    text = CHUNKS.read_text(encoding="utf-8")
    return [json.loads(l) for l in text.split("\n") if l.strip()]

class BgeM3:
    """BGE-M3 run with ONNX Runtime (Microsoft) instead of PyTorch.
    PyTorch is blocked on this laptop by Windows Smart App Control; the official
    ONNX export from BAAI gives the same 1,024-number vectors."""
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        print(f"loading {MODEL} ONNX (first time downloads ~2.3 GB)...", flush=True)
        path = Path(snapshot_download(MODEL, allow_patterns=["onnx/*"])) / "onnx"
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding(pad_id=1, pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False                       # lower memory use
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "4"))
        self.sess = ort.InferenceSession(str(path / "model.onnx"), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def encode(self, texts):
        import numpy as np
        out = []
        for i in range(0, len(texts), BATCH):
            enc = self.tok.encode_batch(texts[i:i + BATCH])
            feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                    "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
            feed = {k: v for k, v in feed.items() if k in self.inputs}
            res = dict(zip(self.outputs, self.sess.run(None, feed)))
            if "dense_vecs" in res:
                vec = res["dense_vecs"]
            else:                                   # CLS token = first token of last layer
                vec = res[self.outputs[0]][:, 0, :]
            out.append(vec)
        v = np.concatenate(out)
        # normalize -> every vector has length 1, so cosine similarity = dot product
        return v / np.linalg.norm(v, axis=1, keepdims=True)

def load_model():
    return BgeM3()

def embed(model, texts):
    return model.encode(texts).tolist()

def metadata(c):
    """Chroma only stores str / int / float / bool values."""
    keep = ("url", "title", "section", "page_type", "language", "location",
            "chunk_type", "chunk_index", "weight")
    return {k: c[k] for k in keep if c.get(k) is not None}


def preview(chunks, pick):
    if pick:
        c = next((c for c in chunks if pick in c["url"]), None)
        if not c:
            sys.exit(f"no chunk with url containing '{pick}'")
    else:
        c = next(c for c in chunks if c["chunk_type"] == "project_card")
    model = load_model()
    t = time.time()
    [vec] = embed(model, [c["text"]])
    print(f"\n===== CHUNK  (id {c['id']}, {len(c['text'])} chars)\n{c['text']}\n")
    print("===== METADATA (stored next to the numbers)")
    for k, v in metadata(c).items():
        print(f"  {k}: {v}")
    print(f"\n===== EMBEDDING  (took {time.time() - t:.1f}s)")
    print(f"  length : {len(vec)} numbers")
    print(f"  first 10: {[round(x, 4) for x in vec[:10]]}")
    print(f"  last  3 : {[round(x, 4) for x in vec[-3:]]}")

    # does it work? compare the chunk with a few questions (1.0 = same meaning, 0 = unrelated)
    name = c["title"]
    questions = [f"Tell me about {name}",
                 f"Where is {name} located?",
                 "ما هي مميزات هذا المشروع؟",            # Arabic: what are this project's features?
                 "What is the weather in London today?"]  # unrelated, should score low
    qv = embed(model, questions)
    print("\n===== TEST: similarity between this chunk and some questions")
    for q, v in zip(questions, qv):
        print(f"  {sum(a * b for a, b in zip(vec, v)):.3f}  {q}")
    print("\npreview only - nothing was saved.")


def build(chunks):
    import chromadb
    client = chromadb.PersistentClient(path=str(DB_DIR))
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    done = set(col.get(include=[])["ids"]) if col.count() else set()
    todo = [c for c in chunks if c["id"] not in done]
    print(f"chunks: {len(chunks)}, already in db: {len(done)}, to embed: {len(todo)}")
    if not todo:
        return
    model = load_model()
    step, start = 32, time.time()
    for i in range(0, len(todo), step):
        part = todo[i:i + step]
        col.add(ids=[c["id"] for c in part],
                documents=[c["text"] for c in part],
                embeddings=embed(model, [c["text"] for c in part]),
                metadatas=[metadata(c) for c in part])
        n = i + len(part)
        rate = n / (time.time() - start)
        print(f"  {n}/{len(todo)}  ({rate:.1f} chunks/s, ~{(len(todo) - n) / rate / 60:.0f} min left)",
              flush=True)
    print(f"done: {col.count()} chunks in {DB_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="embed one chunk and show it")
    ap.add_argument("--pick", help="preview the first chunk whose url contains this text")
    a = ap.parse_args()
    chunks = load_chunks()
    preview(chunks, a.pick) if a.preview else build(chunks)
