"""Batch-summarise all past threads with Haiku → data/thread_summary.md

Run once (and nightly) to keep the thread summary fresh:
    python -m agent.summarize_threads
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
from agent.lumenx_api import get_export
from agent.cost_logger import compute_cost, log_call

_MODEL = "claude-haiku-4-5-20251001"
_SUMMARY_PATH = Path(__file__).parent.parent / "data" / "thread_summary.md"
_MAX_THREADS = 60       # cap to avoid huge prompts
_MAX_MSG_PER_THREAD = 6 # messages sampled per thread
_MAX_MSG_LEN = 220      # chars per message


def _format_thread(thread: dict, idx: int) -> str:
    messages = thread.get("messages", [])
    lines = []
    for m in messages[:_MAX_MSG_PER_THREAD]:
        role = "Customer" if m.get("role") == "customer" else "Support"
        text = (m.get("text") or "").strip()[:_MAX_MSG_LEN]
        if text:
            lines.append(f"  {role}: {text}")
    if not lines:
        return ""
    product = thread.get("product_id") or thread.get("product") or ""
    header = f"Thread {idx}" + (f" [{product}]" if product else "")
    return header + "\n" + "\n".join(lines)


def run() -> None:
    print("Fetching export...")
    export = get_export()

    threads = export.get("threads", [])
    if not threads:
        print("No threads in export.")
        return

    # Only threads that received an admin reply (completed exchanges)
    completed = [
        t for t in threads
        if any(m.get("role") == "admin" for m in t.get("messages", []))
    ]
    print(f"  {len(completed)} completed threads found.")

    sample = completed[-_MAX_THREADS:]
    snippets = []
    for i, t in enumerate(sample, 1):
        snippet = _format_thread(t, i)
        if snippet:
            snippets.append(snippet)

    if not snippets:
        print("No formattable threads.")
        return

    threads_block = "\n\n".join(snippets)

    prompt = f"""You are reviewing {len(snippets)} past customer support exchanges for LumenX, \
a SaaS platform with 20 business tools.

{threads_block}

Write a concise 150-200 word summary covering:
1. The most common customer question types (pricing, features, integrations, billing, etc.)
2. Which LumenX products appear most often
3. The reply tone and style that works best (empathetic, direct, brief)
4. Any recurring confusion or pain points to watch for

This summary is injected into a live support agent's context window before each reply."""

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_MODEL,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )

    summary_text = response.content[0].text.strip()
    cost = compute_cost(_MODEL, response.usage.input_tokens, response.usage.output_tokens)
    log_call(
        thread_id="_summarize_threads",
        model=_MODEL,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cost_usd=cost,
        extra={"action": "thread_summary", "threads_sampled": len(snippets)},
    )

    _SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY_PATH.write_text(summary_text, encoding="utf-8")

    print(f"Summary saved → {_SUMMARY_PATH}")
    print(f"Cost: ${cost:.4f}  (in={response.usage.input_tokens}, out={response.usage.output_tokens})")
    print(f"\n--- Summary ---\n{summary_text}\n")


if __name__ == "__main__":
    run()
