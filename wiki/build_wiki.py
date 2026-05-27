"""Fetch products from LumenX API, chunk into sections, embed, save to wiki_data/.

Usage:
    python -m wiki.build_wiki
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from agent.lumenx_api import get_products
from wiki.wiki_store import WIKI_DATA_DIR


def _stringify(value) -> str:
    if isinstance(value, list):
        return "\n".join(f"  - {item}" for item in value)
    if isinstance(value, dict):
        return json.dumps(value, indent=2)
    return str(value).strip()


def _chunk_product(product: dict) -> list[dict]:
    pid = product.get("id") or product.get("slug") or "unknown"
    pname = product.get("name") or pid
    chunks = []

    def _add(section: str, keys: list[str], label: str) -> None:
        for key in keys:
            val = product.get(key)
            if val:
                chunks.append({
                    "id": f"{pid}__{section}",
                    "product_id": pid,
                    "product_name": pname,
                    "section": section,
                    "text": f"{pname} — {label}:\n{_stringify(val)}",
                })
                return

    # Overview
    overview_parts = []
    for key in ("description", "overview", "summary", "about"):
        if product.get(key):
            overview_parts.append(_stringify(product[key]))
    for key in ("target_audience", "targetAudience", "target_market", "audience"):
        if product.get(key):
            overview_parts.append(f"Target audience: {_stringify(product[key])}")
    if overview_parts:
        chunks.append({
            "id": f"{pid}__overview",
            "product_id": pid,
            "product_name": pname,
            "section": "overview",
            "text": f"{pname} — overview:\n" + "\n".join(overview_parts),
        })

    _add("features",      ["features", "feature_list", "featureList", "key_features"], "features")
    _add("pricing",       ["pricing", "pricing_tiers", "pricingTiers", "plans", "price"], "pricing tiers")
    _add("refund_policy", ["refund_policy", "refundPolicy", "refund", "cancellation",
                           "cancellation_policy", "cancellationPolicy"], "refund & cancellation policy")
    _add("integrations",  ["integrations", "integration_list", "integrationList", "connects_with"], "integrations")
    _add("support_sla",   ["support_sla", "supportSla", "sla", "support_level", "support"], "support SLA")

    return chunks


def build() -> None:
    print("Fetching products from LumenX API...")
    resp = get_products()

    # API may return a list or a dict with a 'products' key
    if isinstance(resp, list):
        products = resp
        policies = None
    else:
        products = resp.get("products") or []
        policies = resp.get("policies") or resp.get("company_policies")

    print(f"  {len(products)} products found.")

    all_chunks: list[dict] = []
    for p in products:
        all_chunks.extend(_chunk_product(p))

    # Company-wide policies as a global chunk
    if policies:
        all_chunks.append({
            "id": "_global__policies",
            "product_id": "_global",
            "product_name": "LumenX",
            "section": "policies",
            "text": f"LumenX company-wide policies:\n{_stringify(policies)}",
        })

    if not all_chunks:
        print("No chunks produced — check API response structure.")
        sys.exit(1)

    # Add positional index
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i

    print(f"  {len(all_chunks)} chunks total. Building TF-IDF index...")

    import pickle
    import scipy.sparse as sp
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=20000,
                                 sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(texts)

    WIKI_DATA_DIR.mkdir(parents=True, exist_ok=True)

    with (WIKI_DATA_DIR / "chunks.json").open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)
    with (WIKI_DATA_DIR / "vectorizer.pkl").open("wb") as f:
        pickle.dump(vectorizer, f)
    sp.save_npz(str(WIKI_DATA_DIR / "tfidf_matrix.npz"), tfidf_matrix)

    print(f"Wiki built: {len(all_chunks)} chunks, "
          f"{tfidf_matrix.shape[1]} TF-IDF features -> {WIKI_DATA_DIR}")


if __name__ == "__main__":
    build()
