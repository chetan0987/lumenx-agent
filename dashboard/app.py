"""Phase 4+5 — Human Review + Confidence Net Dashboard (FastAPI, port 8001).

Usage:
    uvicorn dashboard.app:app --port 8001 --reload
"""

import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

STATIC = Path(__file__).parent / "static"
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))

import json
from collections import defaultdict
from datetime import timedelta

from agent import lumenx_api as api
from agent import llm_draft, context_builder, intent_router
from agent import cost_logger, feedback_log, confidence_net

_COST_LOG_PATH = ROOT / "data" / "cost_log.jsonl"


def _load_cost_log() -> list[dict]:
    if not _COST_LOG_PATH.exists():
        return []
    entries = []
    with _COST_LOG_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries

app = FastAPI(title="LumenX Review Dashboard", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

# ── Shared state ──────────────────────────────────────────────────────────────
_SEEN_PATH = ROOT / "data" / "seen_threads.txt"
_THRESHOLD_PATH = ROOT / "data" / "threshold.txt"

def _load_seen() -> set[str]:
    if _SEEN_PATH.exists():
        return set(_SEEN_PATH.read_text().splitlines())
    return set()

def _persist_seen(seen: set[str]) -> None:
    _SEEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    _SEEN_PATH.write_text("\n".join(seen))

def _load_threshold() -> float:
    if _THRESHOLD_PATH.exists():
        try:
            return float(_THRESHOLD_PATH.read_text().strip())
        except ValueError:
            pass
    return float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))

def _persist_threshold(t: float) -> None:
    _THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    _THRESHOLD_PATH.write_text(str(t))

_queue: dict[str, dict] = {}
_seen_threads: set[str] = _load_seen()
_server_time: Optional[str] = None
_processed_today: int = 0
_auto_sent_today: int = 0
_cost_today: float = 0.0
_threshold: float = _load_threshold()


# ── Background polling ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(_polling_loop())


async def _polling_loop():
    global _server_time
    print("[poller] started")
    while True:
        try:
            inbox = await asyncio.to_thread(api.get_inbox, since=_server_time)
            _server_time = inbox.get("server_time")
            for entry in inbox.get("entries", []):
                thread_meta = entry.get("thread", {})
                thread_id = thread_meta.get("id")
                if not thread_id or not entry.get("awaiting_admin"):
                    continue
                if thread_id in _seen_threads:
                    continue
                _seen_threads.add(thread_id)
                _persist_seen(_seen_threads)
                asyncio.create_task(_process_thread(thread_id))
        except Exception as exc:
            print(f"[poller] error: {exc}")
        await asyncio.sleep(POLL_INTERVAL)


async def _process_thread(thread_id: str):
    global _processed_today, _auto_sent_today, _cost_today
    try:
        thread = await asyncio.to_thread(api.get_thread, thread_id)
        messages = thread.get("messages", [])
        if not messages:
            return

        customer_msgs = [m for m in messages if m.get("role") == "customer"]
        customer_preview = ((customer_msgs[-1].get("text") or "") if customer_msgs else "").strip()[:300]
        thread_turn = len(customer_msgs)

        intent = await asyncio.to_thread(
            intent_router.classify_intent, messages, thread_id
        )

        if intent == "greeting":
            draft = await asyncio.to_thread(llm_draft.generate_greeting_draft, messages)
            kb_hits, chunks, context_str = 0, [], ""
        else:
            ctx_str, kb_hits, chunks = await asyncio.to_thread(
                context_builder.build_context, messages
            )
            draft = await asyncio.to_thread(llm_draft.generate_draft, messages, ctx_str)
            context_str = ctx_str

        item_data = {
            "id": str(uuid.uuid4()),
            "thread_id": thread_id,
            "messages": messages,
            "customer_preview": customer_preview,
            "intent": intent,
            "draft": draft,
            "kb_hits": kb_hits,
            "chunks": chunks,
            "context_str": context_str,
            "thread_turn": thread_turn,
            "arrived_at": datetime.now(timezone.utc).isoformat(),
        }

        # Phase 5: confidence scoring
        confidence = await asyncio.to_thread(confidence_net.score_item, item_data)
        item_data["confidence"] = confidence

        if confidence_net.is_trained() and confidence >= _threshold:
            # Auto-send
            await asyncio.to_thread(
                api.send_reply, thread_id, draft["text"], "agent", confidence
            )
            await asyncio.to_thread(api.mark_read, thread_id)

            context_used = [f"{c['product_name']} / {c['section']}" for c in chunks]
            feedback_log.append_entry({
                "thread_id": thread_id,
                "intent": intent,
                "draft_text": draft["text"],
                "final_text": draft["text"],
                "edit_ratio": 0.0,
                "was_auto_sent": True,
                "human_approved": True,
                "context_used": context_used,
                "confidence_score": confidence,
                "kb_hits": kb_hits,
                "thread_turn": thread_turn,
                "input_tokens": draft["input_tokens"],
                "output_tokens": draft["output_tokens"],
                "model": draft["model"],
                "cost_usd": draft["cost_usd"],
            })
            cost_logger.log_call(
                thread_id=thread_id,
                model=draft["model"],
                input_tokens=draft["input_tokens"],
                output_tokens=draft["output_tokens"],
                cost_usd=draft["cost_usd"],
                extra={"action": "auto_sent", "intent": intent,
                       "confidence": confidence, "kb_hits": kb_hits},
            )
            _auto_sent_today += 1
            _processed_today += 1
            _cost_today += draft["cost_usd"]
            print(f"[auto-send] thread={thread_id} conf={confidence:.3f}")
        else:
            _queue[item_data["id"]] = item_data
            print(f"[queue] +{item_data['id'][:8]} thread={thread_id} "
                  f"intent={intent} conf={confidence:.3f}")

    except Exception as exc:
        print(f"[process] thread={thread_id} error: {exc}")
        import traceback; traceback.print_exc()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def index():
    return FileResponse(STATIC / "review.html")


@app.get("/api/queue")
async def get_queue():
    items = [{
        "id": item["id"],
        "thread_id": item["thread_id"],
        "customer_preview": item["customer_preview"],
        "intent": item["intent"],
        "draft_text": item["draft"]["text"],
        "model": item["draft"]["model"],
        "input_tokens": item["draft"]["input_tokens"],
        "output_tokens": item["draft"]["output_tokens"],
        "cost_usd": item["draft"]["cost_usd"],
        "kb_hits": item["kb_hits"],
        "chunks": item["chunks"],
        "arrived_at": item["arrived_at"],
        "confidence": item.get("confidence", 0.5),
    } for item in _queue.values()]
    items.sort(key=lambda x: x["arrived_at"])
    return {"items": items, "count": len(items)}


@app.get("/api/stats")
async def get_stats():
    return {
        "pending": len(_queue),
        "processed_today": _processed_today,
        "auto_sent_today": _auto_sent_today,
        "cost_today_usd": round(_cost_today, 6),
        "threshold": _threshold,
        "model_trained": confidence_net.is_trained(),
    }


class ActionRequest(BaseModel):
    action: str          # "approve" | "edit" | "skip"
    edited_text: str = ""


@app.post("/api/action/{item_id}")
async def take_action(item_id: str, req: ActionRequest):
    global _processed_today, _cost_today

    item = _queue.get(item_id)
    if not item:
        raise HTTPException(status_code=404, detail="Item not found or already actioned")

    thread_id = item["thread_id"]
    draft = item["draft"]
    intent = item["intent"]

    if req.action == "skip":
        del _queue[item_id]
        return {"status": "skipped"}

    if req.action == "approve":
        final_text = draft["text"]
        draft_source = "agent"
        edit_ratio = 0.0
    elif req.action == "edit":
        if not req.edited_text.strip():
            raise HTTPException(status_code=400, detail="edited_text is required")
        final_text = req.edited_text.strip()
        draft_source = "human"
        d_len = len(draft["text"])
        edit_ratio = round(abs(len(final_text) - d_len) / max(d_len, 1), 4)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {req.action!r}")

    await asyncio.to_thread(api.send_reply, thread_id, final_text, draft_source, None)
    await asyncio.to_thread(api.mark_read, thread_id)

    cost_logger.log_call(
        thread_id=thread_id,
        model=draft["model"],
        input_tokens=draft["input_tokens"],
        output_tokens=draft["output_tokens"],
        cost_usd=draft["cost_usd"],
        extra={
            "action": "sent_as_is" if req.action == "approve" else "edited",
            "draft_source": draft_source,
            "intent": intent,
            "kb_hits": item["kb_hits"],
        },
    )

    context_used = [f"{c['product_name']} / {c['section']}" for c in item.get("chunks", [])]
    feedback_log.append_entry({
        "thread_id": thread_id,
        "intent": intent,
        "draft_text": draft["text"],
        "final_text": final_text,
        "edit_ratio": edit_ratio,
        "was_auto_sent": False,
        "human_approved": True,
        "context_used": context_used,
        "confidence_score": item.get("confidence"),
        "kb_hits": item.get("kb_hits", 0),
        "thread_turn": item.get("thread_turn", 1),
        "input_tokens": draft["input_tokens"],
        "output_tokens": draft["output_tokens"],
        "model": draft["model"],
        "cost_usd": draft["cost_usd"],
    })

    _processed_today += 1
    _cost_today += draft["cost_usd"]
    del _queue[item_id]
    return {"status": req.action, "thread_id": thread_id}


@app.get("/api/feedback/recent")
async def get_recent_feedback():
    entries = feedback_log.load_recent(50)
    return {"entries": list(reversed(entries)), "count": len(entries)}


# ── Analytics endpoint ────────────────────────────────────────────────────────

@app.get("/api/analytics")
async def get_analytics():
    cost_entries = await asyncio.to_thread(_load_cost_log)
    fb_entries   = await asyncio.to_thread(feedback_log.load_all)

    today = datetime.now(timezone.utc).date()

    # ── Summary ────────────────────────────────────────────────────────────────
    total_cost  = sum(e.get("cost_usd", 0) for e in cost_entries)
    total_calls = len(cost_entries)

    fb_decided  = [e for e in fb_entries if "was_auto_sent" in e]
    auto_sent   = sum(1 for e in fb_decided if e.get("was_auto_sent"))
    auto_rate   = round(auto_sent / max(len(fb_decided), 1) * 100, 1)

    conf_scores = [e["confidence_score"] for e in fb_entries
                   if e.get("confidence_score") is not None]
    avg_conf    = round(sum(conf_scores) / max(len(conf_scores), 1), 3) if conf_scores else None

    # ── Daily cost (last 14 days) ──────────────────────────────────────────────
    daily: dict[str, dict] = {}
    for i in range(14):
        d = (today - timedelta(days=i)).isoformat()
        daily[d] = {"cost": 0.0, "calls": 0}
    for e in cost_entries:
        ts = e.get("timestamp", "")
        d  = ts[:10] if ts else ""
        if d in daily:
            daily[d]["cost"]  += e.get("cost_usd", 0)
            daily[d]["calls"] += 1

    # ── Intent distribution ────────────────────────────────────────────────────
    intent_dist: dict[str, int] = defaultdict(int)
    for e in fb_entries:
        intent_dist[e.get("intent", "other")] += 1

    # ── Model usage ────────────────────────────────────────────────────────────
    model_usage: dict[str, dict] = defaultdict(lambda: {"calls": 0, "cost": 0.0})
    for e in cost_entries:
        m = e.get("model", "unknown")
        model_usage[m]["calls"] += 1
        model_usage[m]["cost"]  += e.get("cost_usd", 0)

    # ── Recent calls (last 50, newest first) ──────────────────────────────────
    recent = cost_entries[-50:][::-1]

    return {
        "summary": {
            "total_cost":       round(total_cost, 6),
            "total_calls":      total_calls,
            "auto_rate_pct":    auto_rate,
            "avg_confidence":   avg_conf,
            "feedback_entries": len(fb_entries),
        },
        "daily_cost": [
            {"date": d, "cost": round(v["cost"], 6), "calls": v["calls"]}
            for d, v in sorted(daily.items())
        ],
        "intent_dist": dict(intent_dist),
        "model_usage": {
            k: {"calls": v["calls"], "cost": round(v["cost"], 6)}
            for k, v in model_usage.items()
        },
        "recent_calls": recent,
    }


# ── Confidence Net endpoints ──────────────────────────────────────────────────

class ThresholdRequest(BaseModel):
    threshold: float


@app.get("/api/confidence/status")
async def confidence_status():
    return confidence_net.get_status()


@app.post("/api/confidence/train")
async def train_model():
    result = await asyncio.to_thread(confidence_net.retrain)
    return result


@app.get("/api/confidence/threshold")
async def get_threshold_api():
    return {"threshold": _threshold}


@app.post("/api/confidence/threshold")
async def set_threshold_api(req: ThresholdRequest):
    global _threshold
    if not 0.0 <= req.threshold <= 1.0:
        raise HTTPException(status_code=400, detail="threshold must be 0–1")
    _threshold = round(req.threshold, 2)
    _persist_threshold(_threshold)
    return {"threshold": _threshold}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8001, reload=True)
