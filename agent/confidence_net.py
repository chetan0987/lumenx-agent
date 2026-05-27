"""Phase 5 — Confidence Net: MLP that scores draft reply quality.

Features (8-element vector):
  - intent_type    : one-hot 4-vector (greeting/pricing/technical/other)
  - kb_hits        : KB chunks retrieved, normalised to [0,1]
  - draft_len_norm : draft token-length estimate, normalised to [0,1]
  - thread_turn    : customer message count, normalised to [0,1]
  - fb_match_score : cosine similarity to closest past feedback entry

Label:
  1 = sent as-is  (edit_ratio < 0.01)
  0 = human-edited (edit_ratio >= 0.01)

Architecture: MLP 8->64->64->1  ReLU  sigmoid  (sklearn MLPClassifier)
Min samples:  CONFIDENCE_MIN_SAMPLES env var (default 10; recommend 200+ in prod)
"""

import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from agent.feedback_log import load_all

_MODEL_PATH = Path(__file__).parent.parent / "models" / "confidence_mlp.pkl"
_MIN_SAMPLES = int(os.getenv("CONFIDENCE_MIN_SAMPLES", "10"))
_INTENT_MAP = {"greeting": 0, "pricing": 1, "technical": 2, "other": 3}

_bundle: dict | None = None


def is_trained() -> bool:
    return _MODEL_PATH.exists()


def _load_bundle() -> dict | None:
    global _bundle
    if _bundle is not None:
        return _bundle
    if not _MODEL_PATH.exists():
        return None
    with _MODEL_PATH.open("rb") as f:
        _bundle = pickle.load(f)
    return _bundle


def _fvec(intent: str, kb_hits: int, draft_text: str,
          thread_turn: int, fb_match: float) -> list[float]:
    ih = _INTENT_MAP.get(intent, 3)
    oh = [0.0, 0.0, 0.0, 0.0]
    oh[ih] = 1.0
    kb = min(kb_hits, 64) / 64.0
    dl = min(len(draft_text) / 4.0, 1000.0) / 1000.0
    tt = min(thread_turn, 20) / 20.0
    return oh + [kb, dl, tt, fb_match]


def score_item(item: dict) -> float:
    """Return P(no human edit needed) in [0,1].  Returns 0.5 if not trained."""
    bundle = _load_bundle()
    if bundle is None:
        return 0.5

    intent = item.get("intent", "other")
    kb_hits = item.get("kb_hits", 0)
    draft_text = item.get("draft", {}).get("text", "")
    messages = item.get("messages", [])
    thread_turn = len([m for m in messages if m.get("role") == "customer"])

    fb_matrix = bundle.get("fb_matrix")
    fb_vectorizer = bundle.get("fb_vectorizer")
    if fb_matrix is not None and fb_matrix.shape[0] > 0 and draft_text:
        q = fb_vectorizer.transform([draft_text])
        sims = cosine_similarity(q, fb_matrix)[0]
        fb_match = float(np.max(sims))
    else:
        fb_match = 0.0

    fv = np.array([_fvec(intent, kb_hits, draft_text, thread_turn, fb_match)])
    fv_s = bundle["scaler"].transform(fv)
    prob = float(bundle["clf"].predict_proba(fv_s)[0][1])
    return round(prob, 4)


def retrain() -> dict:
    """Train the MLP from the full feedback log. Returns a status dict."""
    global _bundle

    entries = load_all()
    labeled = [e for e in entries if "final_text" in e and "draft_text" in e]

    if len(labeled) < _MIN_SAMPLES:
        return {
            "status": "insufficient_data",
            "samples": len(labeled),
            "needed": _MIN_SAMPLES,
        }

    labels = np.array([
        1 if e.get("edit_ratio", 1.0) < 0.01 else 0
        for e in labeled
    ])

    unique = np.unique(labels)
    if len(unique) < 2:
        return {
            "status": "error",
            "message": f"Only class {int(unique[0])} in training data — need both 0 and 1.",
            "samples": len(labeled),
        }

    fb_texts = [e.get("draft_text", "") for e in labeled]
    fb_vect = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    fb_mat = fb_vect.fit_transform(fb_texts)

    # Leave-one-out fb_match to avoid self-similarity data leak
    rows = []
    for i, e in enumerate(labeled):
        mask = np.arange(len(labeled)) != i
        if mask.any():
            sims = cosine_similarity(fb_mat[i], fb_mat[mask])[0]
            fb_match = float(np.max(sims))
        else:
            fb_match = 0.0
        rows.append(_fvec(
            e.get("intent", "other"),
            e.get("kb_hits", 0),
            e.get("draft_text", ""),
            e.get("thread_turn", 1),
            fb_match,
        ))

    X = np.array(rows)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    clf = MLPClassifier(
        hidden_layer_sizes=(64, 64),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=len(labeled) >= 20,
    )
    clf.fit(X_s, labels)
    train_acc = float(clf.score(X_s, labels))

    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_bundle = {"clf": clf, "scaler": scaler,
                  "fb_vectorizer": fb_vect, "fb_matrix": fb_mat}
    with _MODEL_PATH.open("wb") as f:
        pickle.dump(new_bundle, f)
    _bundle = new_bundle

    return {
        "status": "trained",
        "samples": len(labeled),
        "train_accuracy": round(train_acc, 4),
        "label_dist": {
            "sent_as_is": int(sum(labels)),
            "edited": int(len(labels) - sum(labels)),
        },
    }


def get_status() -> dict:
    entries = load_all()
    labeled = [e for e in entries if "final_text" in e and "draft_text" in e]
    trained = is_trained()
    result = {
        "trained": trained,
        "samples": len(labeled),
        "needed": _MIN_SAMPLES,
        "ready_to_train": len(labeled) >= _MIN_SAMPLES,
    }
    if trained:
        b = _load_bundle()
        if b and "clf" in b:
            result["n_iter"] = int(getattr(b["clf"], "n_iter_", 0))
    return result
