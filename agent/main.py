"""Phase 3+5 — polling loop with intent routing, confidence scoring, and CLI review."""

import os
import sys
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

from agent import lumenx_api as api
from agent import llm_draft, cost_logger, context_builder, intent_router, confidence_net
from agent import feedback_log

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "5"))
THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))
LINE = "─" * 64

_INTENT_ICON = {
    "greeting":  "👋",
    "pricing":   "💰",
    "technical": "🔧",
    "other":     "💬",
}


def _format_thread_preview(thread: dict) -> str:
    msgs = thread.get("messages", [])
    customer_msgs = [m for m in msgs if m.get("role") == "customer"]
    last = customer_msgs[-1] if customer_msgs else {}
    return (last.get("text") or "").strip()[:200]


def _cli_review(thread_id: str, customer_preview: str, draft: dict,
                intent: str = "other", kb_hits: int = 0,
                confidence: float = 0.5) -> None:
    icon = _INTENT_ICON.get(intent, "💬")
    trained_marker = f"  conf={confidence:.2f}" if confidence_net.is_trained() else "  conf=— (not trained)"
    print(f"\n{LINE}")
    print(f"Thread  : {thread_id}")
    print(f"Intent  : {icon} {intent}  |  KB hits: {kb_hits}{trained_marker}")
    print(f"Customer: {customer_preview}")
    print(LINE)
    print(f"DRAFT   : {draft['text']}")
    print(LINE)
    print(f"Tokens  : in={draft['input_tokens']}  out={draft['output_tokens']}  "
          f"cost=${draft['cost_usd']:.4f}  model={draft['model']}")
    print(LINE)

    while True:
        choice = input("[S]end  [E]dit  [K]ip  [Q]uit > ").strip().lower()

        if choice == "s":
            api.send_reply(thread_id, draft["text"],
                           draft_source="agent", confidence=None)
            feedback_log.append_entry({
                "thread_id": thread_id,
                "intent": intent,
                "draft_text": draft["text"],
                "final_text": draft["text"],
                "edit_ratio": 0.0,
                "was_auto_sent": False,
                "human_approved": True,
                "confidence_score": confidence if confidence_net.is_trained() else None,
                "kb_hits": kb_hits,
                "input_tokens": draft["input_tokens"],
                "output_tokens": draft["output_tokens"],
                "model": draft["model"],
                "cost_usd": draft["cost_usd"],
            })
            cost_logger.log_call(
                thread_id=thread_id, model=draft["model"],
                input_tokens=draft["input_tokens"], output_tokens=draft["output_tokens"],
                cost_usd=draft["cost_usd"],
                extra={"action": "sent_as_is", "intent": intent, "kb_hits": kb_hits,
                       "confidence": confidence},
            )
            api.mark_read(thread_id)
            print("  Sent.")
            break

        elif choice == "e":
            print("  Enter your reply (finish with a blank line):")
            lines = []
            while True:
                ln = input()
                if ln == "":
                    break
                lines.append(ln)
            edited_text = "\n".join(lines).strip()
            if edited_text:
                api.send_reply(thread_id, edited_text,
                               draft_source="human", confidence=None)
                d_len = len(draft["text"])
                edit_ratio = round(abs(len(edited_text) - d_len) / max(d_len, 1), 4)
                feedback_log.append_entry({
                    "thread_id": thread_id,
                    "intent": intent,
                    "draft_text": draft["text"],
                    "final_text": edited_text,
                    "edit_ratio": edit_ratio,
                    "was_auto_sent": False,
                    "human_approved": True,
                    "confidence_score": confidence if confidence_net.is_trained() else None,
                    "kb_hits": kb_hits,
                    "input_tokens": draft["input_tokens"],
                    "output_tokens": draft["output_tokens"],
                    "model": draft["model"],
                    "cost_usd": draft["cost_usd"],
                })
                cost_logger.log_call(
                    thread_id=thread_id, model=draft["model"],
                    input_tokens=draft["input_tokens"], output_tokens=draft["output_tokens"],
                    cost_usd=draft["cost_usd"],
                    extra={"action": "edited", "intent": intent, "kb_hits": kb_hits,
                           "edit_ratio": edit_ratio, "confidence": confidence},
                )
                api.mark_read(thread_id)
                print("  Sent (edited).")
            else:
                print("  Empty reply — skipped.")
            break

        elif choice == "k":
            print("  Skipped.")
            break

        elif choice == "q":
            totals = cost_logger.session_total()
            print(f"\nSession totals: calls={totals['calls']}  "
                  f"tokens={totals['input_tokens']+totals['output_tokens']}  "
                  f"cost=${totals['cost_usd']:.4f}")
            sys.exit(0)

        else:
            print("  Unknown — use S, E, K, or Q.")


def run() -> None:
    print("LumenX Auto-Reply Agent — Phase 3+5")
    status = confidence_net.get_status()
    if status["trained"]:
        print(f"Confidence Net: trained ({status['samples']} samples)  "
              f"threshold={THRESHOLD:.0%}")
    else:
        print(f"Confidence Net: not trained yet "
              f"({status['samples']}/{status['needed']} samples)  "
              f"— all to human review")
    print(f"Polling every {POLL_INTERVAL}s. Press Ctrl-C to stop.\n")

    server_time: str | None = None
    seen: set[str] = set()

    while True:
        try:
            inbox = api.get_inbox(since=server_time)
            server_time = inbox.get("server_time")
            entries = inbox.get("entries", [])

            for entry in entries:
                thread_meta = entry.get("thread", {})
                thread_id = thread_meta.get("id")

                if not thread_id or not entry.get("awaiting_admin"):
                    continue
                if thread_id in seen:
                    continue
                seen.add(thread_id)

                thread = api.get_thread(thread_id)
                messages = thread.get("messages", [])
                if not messages:
                    continue

                customer_preview = _format_thread_preview(thread)
                print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                      f"New message on {thread_id}")

                intent = "other"
                kb_hits = 0
                confidence = 0.5
                draft = None

                try:
                    intent = intent_router.classify_intent(messages, thread_id=thread_id)
                    print(f"  Intent: {_INTENT_ICON.get(intent,'?')} {intent}")

                    if intent == "greeting":
                        draft = llm_draft.generate_greeting_draft(messages)
                        kb_hits = 0
                    else:
                        ctx_str, kb_hits, _ = context_builder.build_context(messages)
                        draft = llm_draft.generate_draft(messages, context_str=ctx_str)

                    # Phase 5: confidence scoring
                    confidence = confidence_net.score_item({
                        "intent": intent,
                        "kb_hits": kb_hits,
                        "draft": draft,
                        "messages": messages,
                    })
                    print(f"  Confidence: {confidence:.3f}  threshold={THRESHOLD:.0%}")

                except Exception as exc:
                    print(f"  LLM error: {exc}")
                    continue

                # Auto-send if model trained and above threshold
                if confidence_net.is_trained() and confidence >= THRESHOLD:
                    api.send_reply(thread_id, draft["text"], "agent", confidence)
                    feedback_log.append_entry({
                        "thread_id": thread_id,
                        "intent": intent,
                        "draft_text": draft["text"],
                        "final_text": draft["text"],
                        "edit_ratio": 0.0,
                        "was_auto_sent": True,
                        "human_approved": True,
                        "confidence_score": confidence,
                        "kb_hits": kb_hits,
                        "input_tokens": draft["input_tokens"],
                        "output_tokens": draft["output_tokens"],
                        "model": draft["model"],
                        "cost_usd": draft["cost_usd"],
                    })
                    cost_logger.log_call(
                        thread_id=thread_id, model=draft["model"],
                        input_tokens=draft["input_tokens"], output_tokens=draft["output_tokens"],
                        cost_usd=draft["cost_usd"],
                        extra={"action": "auto_sent", "intent": intent,
                               "confidence": confidence, "kb_hits": kb_hits},
                    )
                    api.mark_read(thread_id)
                    print(f"  [AUTO-SENT] conf={confidence:.3f} >= threshold={THRESHOLD:.0%}")
                    continue

                _cli_review(thread_id, customer_preview, draft,
                            intent=intent, kb_hits=kb_hits, confidence=confidence)

        except KeyboardInterrupt:
            totals = cost_logger.session_total()
            print(f"\n\nStopped. Session totals: calls={totals['calls']}  "
                  f"tokens={totals['input_tokens']+totals['output_tokens']}  "
                  f"cost=${totals['cost_usd']:.4f}")
            break
        except Exception as exc:
            print(f"  Poll error: {exc}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run()
