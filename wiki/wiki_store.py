"""TF-IDF based wiki store — fully local, no external model downloads."""

import json
import pickle
import numpy as np
import scipy.sparse as sp
from pathlib import Path

WIKI_DATA_DIR = Path(__file__).parent / "wiki_data"

_vectorizer = None
_tfidf_matrix = None
_chunks: list[dict] | None = None


def _load() -> None:
    global _vectorizer, _tfidf_matrix, _chunks
    if _chunks is not None:
        return

    chunks_path  = WIKI_DATA_DIR / "chunks.json"
    vec_path     = WIKI_DATA_DIR / "vectorizer.pkl"
    matrix_path  = WIKI_DATA_DIR / "tfidf_matrix.npz"

    if not all(p.exists() for p in [chunks_path, vec_path, matrix_path]):
        raise FileNotFoundError("Wiki not built yet. Run:  python -m wiki.build_wiki")

    with chunks_path.open(encoding="utf-8") as f:
        _chunks = json.load(f)
    with vec_path.open("rb") as f:
        _vectorizer = pickle.load(f)
    _tfidf_matrix = sp.load_npz(str(matrix_path))


def is_ready() -> bool:
    return all((WIKI_DATA_DIR / n).exists()
               for n in ["chunks.json", "vectorizer.pkl", "tfidf_matrix.npz"])


def search(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks most relevant to query using TF-IDF cosine similarity."""
    _load()
    from sklearn.metrics.pairwise import cosine_similarity
    q_vec  = _vectorizer.transform([query])
    scores = cosine_similarity(q_vec, _tfidf_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        chunk = dict(_chunks[idx])
        chunk["score"] = float(scores[idx])
        results.append(chunk)
    return results


def chunk_count() -> int:
    if not is_ready():
        return 0
    with (WIKI_DATA_DIR / "chunks.json").open(encoding="utf-8") as f:
        return len(json.load(f))
