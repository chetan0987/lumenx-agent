"""Append-only feedback log — ground truth for Phase 5 Confidence Net training."""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

_LOG_PATH = Path(__file__).parent.parent / "data" / "feedback_log.jsonl"


def append_entry(entry: dict) -> str:
    """Append one feedback entry and return its id."""
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if "id" not in entry:
        entry["id"] = str(uuid.uuid4())
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["id"]


def load_recent(n: int = 50) -> list[dict]:
    """Return the most recent n entries (oldest first)."""
    if not _LOG_PATH.exists():
        return []
    entries = []
    with _LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries[-n:]


def load_all() -> list[dict]:
    return load_recent(n=1_000_000)
