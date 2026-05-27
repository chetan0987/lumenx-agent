"""Phase 3 — Intent Router using Haiku (fast, cheap classification)."""

import json
import os
import re

import anthropic

from agent.cost_logger import compute_cost, log_call

_MODEL = "claude-haiku-4-5-20251001"
_VALID = {"greeting", "pricing", "technical", "other"}

_SYSTEM = (
    "You classify the intent of customer support messages for LumenX, a SaaS platform.\n"
    "Respond with ONLY valid JSON: {\"intent\": \"<label>\"}\n\n"
    "Labels (pick exactly one):\n"
    "  greeting  — hello, hi, thanks, bye, casual small-talk, no product question\n"
    "  pricing   — asking about price, cost, plan, subscription, billing, refund, discount, trial\n"
    "  technical — asking how to use a feature, reporting a bug/error, integration, setup, account issue\n"
    "  other     — feedback, complaints, unclear requests, anything not covered above"
)

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def classify_intent(messages: list[dict], thread_id: str = "_intent") -> str:
    """Return the intent of the last customer message.

    Returns one of: 'greeting', 'pricing', 'technical', 'other'.
    Falls back to 'other' on any error so the pipeline never blocks.
    """
    last_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "customer":
            last_msg = (msg.get("text") or "").strip()
            break

    if not last_msg:
        return "other"

    try:
        response = _get_client().messages.create(
            model=_MODEL,
            max_tokens=32,
            system=_SYSTEM,
            messages=[{"role": "user", "content": last_msg}],
        )
        raw = response.content[0].text.strip()
        # Strip markdown code fences Haiku sometimes adds
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        intent = json.loads(raw).get("intent", "other").lower()
        if intent not in _VALID:
            intent = "other"

        cost = compute_cost(_MODEL, response.usage.input_tokens, response.usage.output_tokens)
        log_call(
            thread_id=thread_id,
            model=_MODEL,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cost_usd=cost,
            extra={"action": "intent_classification", "intent": intent},
        )
        return intent

    except Exception:
        return "other"
