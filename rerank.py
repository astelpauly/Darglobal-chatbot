"""Reranker: takes top-20 chunks from retrieval, scores how well each answers
the question, keeps the best N.

Uses BAAI/bge-reranker-v2-m3 via ONNX Runtime (multilingual, supports Arabic).
Loaded lazily: if the model isn't available, reranking is skipped and the
retrieval order is kept.

The reranker is a 'cross-encoder': it sees (question, chunk) TOGETHER, so it
is slower than embedding, but more accurate. We only run it on 20 chunks.
"""
import os
import numpy as np
from pathlib import Path

# jina-reranker-v2 is multilingual (100+ languages incl. Arabic),
# has an official ONNX export, and is ~550 MB (vs 2.3 GB for bge-reranker-v2-m3).
MODEL = "jinaai/jina-reranker-v2-base-multilingual"
MAX_TOKENS = 512


class Reranker:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from tokenizers import Tokenizer
        print(f"loading {MODEL} ONNX (first time downloads ~550 MB)...", flush=True)
        path = Path(snapshot_download(
            MODEL,
            allow_patterns=["onnx/model.onnx", "onnx/model.onnx_data",
                            "tokenizer.json", "*.json"]))
        onnx_dir = path / "onnx" if (path / "onnx").exists() else path
        onnx_files = list(onnx_dir.glob("model.onnx")) or list(onnx_dir.glob("*.onnx"))
        if not onnx_files:
            raise FileNotFoundError("no .onnx file found for reranker")
        self.tok = Tokenizer.from_file(str(path / "tokenizer.json"))
        self.tok.enable_truncation(MAX_TOKENS)
        self.tok.enable_padding(pad_id=0, pad_token="<pad>")
        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = int(os.getenv("ORT_THREADS", "4"))
        self.sess = ort.InferenceSession(str(onnx_files[0]), opts,
                                         providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def score(self, question, texts, batch=4):
        """One score per (question, text) pair. Higher = better match."""
        scores = []
        for i in range(0, len(texts), batch):
            # the tokenizer joins the pair with its own [SEP] when given two inputs
            enc = self.tok.encode_batch([(question, t) for t in texts[i:i + batch]])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
            feed = {k: v for k, v in feed.items() if k in self.inputs}
            logits = self.sess.run(None, feed)[0]
            s = logits[:, 0] if logits.ndim == 2 and logits.shape[1] == 1 else \
                logits.squeeze() if logits.ndim == 2 else logits
            scores.extend(np.atleast_1d(s).tolist())
        return scores

    def rerank(self, question, chunks, min_n=3, max_n=10,
               prob_threshold=0.4, drop_ratio=0.5):
        """Dynamic top-N: pick as many chunks as actually match the question.

        Rules (applied in order):
        - score every chunk with the cross-encoder
        - convert raw logits -> probability (sigmoid), easy to threshold
        - always keep the best `min_n` chunks (never fewer, even for one-fact questions)
        - add any extra chunk whose probability is >= `prob_threshold`
          AND whose probability is >= `drop_ratio` * top probability (no big drop-off)
        - cap at `max_n` so the LLM context is never flooded
        """
        if not chunks:
            return []
        logits = self.score(question, [c["text"] for c in chunks])
        probs = 1.0 / (1.0 + np.exp(-np.array(logits)))     # sigmoid -> 0..1
        for c, l, p in zip(chunks, logits, probs):
            c["rerank_score"] = float(l)
            c["rerank_prob"]  = float(p)
        chunks.sort(key=lambda c: c["rerank_prob"], reverse=True)

        top_p = chunks[0]["rerank_prob"]
        cutoff = max(prob_threshold, top_p * drop_ratio)
        kept = [c for c in chunks if c["rerank_prob"] >= cutoff]
        n = max(min_n, min(len(kept), max_n))
        return chunks[:n]


def try_load():
    """Return a Reranker or None if it couldn't be loaded. The chatbot keeps
    working either way; reranking just becomes a no-op."""
    try:
        return Reranker()
    except Exception as e:
        print(f"reranker not available ({e}); skipping rerank step")
        return None
