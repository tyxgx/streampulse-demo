"""Vector retrieval for the StreamPulse demo on a 512 MB host.

Production uses Postgres+pgvector. Here the ~216K yearly-grain chunk embeddings are an int8
numpy matrix (83 MB, memory-mapped), scanned in blocks with exact L2 distance; chunk text
lives in SQLite and is read only for the top-k rows. The query embedder is the ONNX build of
the same all-MiniLM-L6-v2 model (fastembed) instead of PyTorch, to stay within RAM.
"""
import json
import sqlite3
from pathlib import Path

import numpy as np

DATA = Path(__file__).parent / "data"
NO_MATCH_DISTANCE_THRESHOLD = 0.95
NO_DATA_REPLY = "I don't have data to answer that."
BLOCK = 20_000

_meta = json.load(open(DATA / "meta.json"))
_scale = np.float32(_meta["scale"])
_matrix = np.load(DATA / "chunks_q8.npy", mmap_mode="r")
_norms_sq = np.load(DATA / "norms_sq.npy")
_db = sqlite3.connect(DATA / "chunks.sqlite", check_same_thread=False)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


def embed_query(text):
    return np.asarray(next(iter(_get_embedder().embed([text]))), dtype=np.float32)


def _rows(ids, dists):
    out = []
    for i, d in zip(ids, dists):
        t, k, txt = _db.execute(
            "SELECT source_table, source_key, chunk_text FROM chunks WHERE id=?", (int(i),)
        ).fetchone()
        out.append({"source_table": t, "source_key": k, "chunk_text": txt, "distance": float(d)})
    return out


def search_chunks(q, top_k=5):
    """Exact L2 nearest neighbours over all chunks (block-wise so temporaries stay ~30 MB)."""
    q2 = float(q @ q)
    dist = np.empty(len(_matrix), dtype=np.float32)
    for s in range(0, len(_matrix), BLOCK):
        blk = _matrix[s:s + BLOCK].astype(np.float32)
        dist[s:s + BLOCK] = _norms_sq[s:s + BLOCK] + q2 - 2.0 * (blk @ q) / _scale
    top = np.argpartition(dist, top_k)[:top_k]
    top = top[np.argsort(dist[top])]
    return _rows(top, np.sqrt(np.maximum(dist[top], 0)))


def search_chunks_for_entity(q, source_table, entity_prefix, top_k=5):
    """Nearest chunks among one entity's own chunks (e.g. a named country)."""
    ids = [r[0] for r in _db.execute(
        "SELECT id FROM chunks WHERE source_table=? AND source_key LIKE ?",
        (source_table, f"{entity_prefix}|%"))]
    if not ids:
        return []
    ids = np.asarray(ids)
    sub = _matrix[ids].astype(np.float32) / _scale
    d = np.linalg.norm(sub - q, axis=1)
    order = np.argsort(d)[:top_k]
    return _rows(ids[order], d[order])


def top1_distance(chunks):
    return chunks[0]["distance"] if chunks else None
