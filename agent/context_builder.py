"""Assemble LLM context from wiki, thread summary, and thread history."""

import os
from pathlib import Path

_SUMMARY_PATH = Path(__file__).parent.parent / "data" / "thread_summary.md"
_TOP_K = int(os.getenv("WIKI_TOP_K", "5"))


def _last_customer_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "customer":
            return (msg.get("text") or "").strip()
    return ""


def _load_thread_summary() -> str | None:
    if _SUMMARY_PATH.exists():
        text = _SUMMARY_PATH.read_text(encoding="utf-8").strip()
        return text if text else None
    return None


def build_context(messages: list[dict]) -> tuple[str, int, list[dict]]:
    """Return (context_str, kb_hits, chunks) to inject into the system prompt.

    chunks is the raw list of retrieved wiki dicts (used by the review UI
    to show which KB sections were used, and by Phase 5 feature extraction).
    """
    parts: list[str] = []
    kb_hits = 0
    chunks: list[dict] = []

    # 1. Product knowledge from LLM Wiki
    try:
        from wiki.wiki_store import search, is_ready
        if is_ready():
            query = _last_customer_message(messages)
            if query:
                chunks = search(query, top_k=_TOP_K)
                kb_hits = len(chunks)
                if chunks:
                    kb_text = "\n\n".join(
                        f"[{c['product_name']} / {c['section']}]\n{c['text']}"
                        for c in chunks
                    )
                    parts.append(f"## PRODUCT KNOWLEDGE\n{kb_text}")
    except Exception:
        pass  # wiki not yet built — proceed without it

    # 2. Past thread summary
    summary = _load_thread_summary()
    if summary:
        parts.append(f"## PAST THREAD SUMMARY\n{summary}")

    return "\n\n".join(parts), kb_hits, chunks
