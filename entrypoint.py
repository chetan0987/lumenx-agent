"""Docker/Railway entrypoint.

Builds the LLM Wiki if not present, then starts the review dashboard.
PORT env var is respected (Railway sets it automatically).
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

# ── Build wiki if not already present ────────────────────────────────────────
wiki_chunks = ROOT / "wiki" / "wiki_data" / "chunks.json"
if not wiki_chunks.exists():
    print("[startup] LLM Wiki not found — building from LumenX API...", flush=True)
    result = subprocess.run(
        [sys.executable, "-m", "wiki.build_wiki"],
        cwd=str(ROOT),
        check=False,
    )
    if result.returncode != 0:
        print("[startup] Wiki build failed — /api/query will return 503 until wiki is built.", flush=True)
    else:
        print("[startup] Wiki built successfully.", flush=True)
else:
    print("[startup] LLM Wiki found — skipping build.", flush=True)

# ── Start the review dashboard ────────────────────────────────────────────────
import uvicorn

port = int(os.environ.get("PORT", "8001"))
print(f"[startup] Starting LumenX Review Dashboard on port {port}...", flush=True)
uvicorn.run("dashboard.app:app", host="0.0.0.0", port=port)
