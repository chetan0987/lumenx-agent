import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Prices per 1M tokens (USD)
_PRICES = {
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.00},
}

_LOG_PATH = Path(__file__).parent.parent / "data" / "cost_log.jsonl"


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price = _PRICES.get(model, {"input": 3.00, "output": 15.00})
    return round(
        (input_tokens / 1_000_000) * price["input"] +
        (output_tokens / 1_000_000) * price["output"],
        6
    )


def log_call(thread_id: str, model: str,
             input_tokens: int, output_tokens: int,
             cost_usd: float, extra: dict = None) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
    }
    if extra:
        entry.update(extra)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def session_total() -> dict:
    """Return total tokens and cost across all logged calls."""
    if not _LOG_PATH.exists():
        return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            totals["calls"] += 1
            totals["input_tokens"] += row.get("input_tokens", 0)
            totals["output_tokens"] += row.get("output_tokens", 0)
            totals["cost_usd"] += row.get("cost_usd", 0.0)
    totals["cost_usd"] = round(totals["cost_usd"], 6)
    return totals
