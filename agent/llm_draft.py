import os
import anthropic
from agent.cost_logger import compute_cost

_MODEL = "claude-sonnet-4-6"
_HAIKU_MODEL = "claude-haiku-4-5-20251001"

_GREETING_SYSTEM = (
    "You are a friendly customer support agent for LumenX, a SaaS productivity platform. "
    "The customer has sent a greeting or casual message. "
    "Reply warmly and briefly (1–2 sentences) and invite them to share what they need help with. "
    "Sign off as \"The LumenX Support Team\" — never use a personal name."
)

_SYSTEM_PROMPT = """You are a professional and empathetic customer support agent for LumenX, \
a SaaS platform offering 20 business productivity tools (e.g. EmailPilot, InvoiceFlow, TaskGrid).

Guidelines:
- Be warm, professional, and concise (2–4 sentences unless complexity demands more).
- NEVER state or imply specific pricing amounts, refund windows, or cancellation terms \
unless they are explicitly provided to you in the conversation context. \
If you lack that information, say exactly: \
"I don't have that specific information right now — our team will follow up shortly."
- Do not invent features, integrations, or policies.
- When stating a fact about a product, mention the product name.
- If the customer seems frustrated, acknowledge it with empathy before answering.
- Sign off as "The LumenX Support Team" — never use a personal name."""

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _build_messages(thread_messages: list[dict]) -> list[dict]:
    """Convert LumenX thread messages to Claude's user/assistant format.

    LumenX roles: 'customer' → 'user', 'admin' → 'assistant'.
    Skips empty messages. Ensures the list ends on a 'user' turn.
    """
    result = []
    for msg in thread_messages:
        role = msg.get("role", "")
        text = (msg.get("text") or "").strip()
        if not text:
            continue
        if role == "customer":
            result.append({"role": "user", "content": text})
        elif role == "admin":
            result.append({"role": "assistant", "content": text})

    # Claude requires the last message to be from the user
    if result and result[-1]["role"] != "user":
        result = result[:-1]

    return result or [{"role": "user", "content": "(no message)"}]


def generate_draft(thread_messages: list[dict], context_str: str = "") -> dict:
    """Generate a candidate reply for a thread.

    context_str is assembled by context_builder and injected after the base
    system prompt so the model has product knowledge, past-thread summary, etc.

    Returns:
        {
            "text": str,
            "model": str,
            "input_tokens": int,
            "output_tokens": int,
            "cost_usd": float,
        }
    """
    messages = _build_messages(thread_messages)
    system = (_SYSTEM_PROMPT + "\n\n" + context_str) if context_str else _SYSTEM_PROMPT
    response = _get_client().messages.create(
        model=_MODEL,
        max_tokens=512,
        system=system,
        messages=messages,
    )

    text = response.content[0].text.strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = compute_cost(_MODEL, input_tokens, output_tokens)

    return {
        "text": text,
        "model": _MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }


def generate_greeting_draft(thread_messages: list[dict]) -> dict:
    """Fast, cheap Haiku reply for greeting/small-talk threads (no KB needed)."""
    messages = _build_messages(thread_messages)
    response = _get_client().messages.create(
        model=_HAIKU_MODEL,
        max_tokens=128,
        system=_GREETING_SYSTEM,
        messages=messages,
    )
    text = response.content[0].text.strip()
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = compute_cost(_HAIKU_MODEL, input_tokens, output_tokens)
    return {
        "text": text,
        "model": _HAIKU_MODEL,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
