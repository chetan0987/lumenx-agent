# CLAUDE.md — LumenX Auto-Reply Agent

## Project Overview

This project builds a fully-automated, human-in-the-loop **Auto-Reply LLM Agent** for the LumenX SaaS platform. The agent answers customer support questions by routing intent, assembling rich context, drafting a reply with Claude, scoring it with a tiny neural network, and either auto-sending or routing to human review.

Deployed platform: https://lumenx-demo.up.railway.app  
Admin UI: https://lumenx-demo.up.railway.app/admin  
GitHub repo: https://github.com/VizuaraAI/lumenx

---

## Architecture (Four-Part Pipeline)

```
Incoming message
      │
      ▼
[1] Intent Router          ← Haiku (cheap, fast)
      │  greeting / pricing / technical / other
      ▼
[2] Context Builder        ← assembles all sources
      │  • Products JSON (from /api/admin/products)
      │  • LLM Wiki (structured KB, Karpathy-style)
      │  • Current thread history
      │  • Summary of all past threads
      │  • Feedback log (similar past Q&A pairs)
      ▼
[3] LLM Draft              ← Claude Sonnet (quality reply)
      │  produces candidate reply
      ▼
[4] Confidence Net         ← tiny MLP (local, fast)
      │  score 0–1
      ├─ score ≥ threshold → auto-send
      └─ score < threshold → human review queue
                                    │
                              [Human edits / approves]
                                    │
                              Feedback log updated
```

---

## API Credentials

```
BASE_URL  = https://lumenx-demo.up.railway.app
ADMIN_TOKEN = lmx_GQlch0Q5NOwVuVSADXRuFNJvxIpzVGwI   # X-Admin-Token header
```

### Key Endpoints Used by the Agent

| Purpose | Endpoint |
|---|---|
| Poll for new messages | `GET /api/admin/inbox?since=<ISO>` |
| Fetch thread detail | `GET /api/admin/threads/{id}` |
| Send reply | `POST /api/admin/threads/{id}/reply` |
| All products + policies | `GET /api/admin/products` |
| Full export for training | `GET /api/admin/export` |
| Mark thread read | `POST /api/admin/threads/{id}/mark-read` |

Reply body schema: `{ "text": "...", "draft_source": "agent"|"human", "confidence": 0.87 }`

---

## LLM Model Strategy (cost-aware)

| Task | Model | Reason |
|---|---|---|
| Intent classification | `claude-haiku-4-5-20251001` | Low cost, fast |
| Reply generation | `claude-sonnet-4-6` | Quality matters |
| Summarisation of past threads | `claude-haiku-4-5-20251001` | Batch, low cost |

All calls must log: `input_tokens`, `output_tokens`, `model`, `cost_usd`, `thread_id`, `timestamp`.

---

## Confidence Net (MLP)

**Features (input vector per draft):**
- `edit_ratio`: (len(final) - len(draft)) / len(draft) — from training data
- `intent_type`: one-hot (greeting=0, pricing=1, technical=2, other=3)
- `kb_hits`: number of relevant KB chunks retrieved (0–64)
- `draft_len_tokens`: length of the draft
- `thread_turn`: which turn in the conversation
- `feedback_match_score`: cosine similarity to closest feedback log entry

**Architecture:** 2 hidden layers × 64 units, ReLU, sigmoid output  
**Loss:** Binary cross entropy  
**Label:** 1 = reply was sent as-is (no human edit), 0 = human edited or rejected  
**Training trigger:** After 200+ labeled examples accumulate in feedback log

**Ground truth collection plan:** During Phases 1–4 (all replies go through human review), every accepted-as-is reply → label 1, every edited reply → label 0. Edit ratio also stored for regression variant.

---

## LLM Wiki

Based on Karpathy's gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f  
A structured, chunked knowledge base of all 20 LumenX products.  
Each chunk: `{ id, product_id, section, text, embedding }`.  
Embeddings stored locally (FAISS or simple cosine search).  
Rebuilt nightly from `/api/admin/products` export.

---

## System Prompt Constraints (MUST follow)

1. **Never hallucinate pricing or refund policy.** If unknown: "I don't have that specific information — our team will follow up shortly."
2. **Always be professional and empathetic.**
3. **If intent is greeting/small-talk:** respond directly without product context.
4. **Cite source** (product name, policy section) when stating a fact.
5. **Keep replies concise** — SaaS support, not essays.

---

## Feedback Log Schema

```json
{
  "id": "uuid",
  "thread_id": "...",
  "intent": "pricing",
  "draft_text": "...",
  "final_text": "...",
  "edit_ratio": 0.12,
  "was_auto_sent": false,
  "human_approved": true,
  "timestamp": "ISO",
  "context_used": ["product:emailpilot", "policy:refund"],
  "confidence_score": 0.73,
  "input_tokens": 1240,
  "output_tokens": 180,
  "model": "claude-sonnet-4-6",
  "cost_usd": 0.0021
}
```

---

## Phased Execution Plan

### Phase 1 — Foundation (API + Basic Agent)
- Scaffold Python project (FastAPI backend + simple frontend)
- Integrate LumenX API: poll inbox, fetch threads, send replies
- Basic Claude Sonnet reply agent (no context beyond thread history)
- Token + cost logging for every call
- All replies go to human review (CLI or simple web UI)

### Phase 2 — Context Builder
- Fetch and cache `/api/admin/products` → Products JSON
- Build LLM Wiki: chunk, embed, and store product knowledge
- Load current thread + retrieve top-K similar KB chunks into context
- Summarise all past threads (batch job, runs nightly)
- Inject summary + feedback log into system prompt

### Phase 3 — Intent Router
- Fine-prompt Haiku classifier: greeting / pricing / technical / other
- If greeting: bypass context builder, reply directly
- Route non-greeting to full pipeline

### Phase 4 — Human Review UI + Feedback Loop
- Web dashboard: inbox list, draft preview, edit box, approve/reject buttons
- On approve-as-is: log label=1 to feedback log
- On edit+send: log label=0, store edit_ratio
- Show confidence score once Phase 5 is live
- Expandable context window per reply (what was injected)

### Phase 5 — Confidence Net
- Feature extraction from feedback log
- Train MLP once 200+ labeled examples exist
- Expose threshold slider in dashboard
- Router: score ≥ threshold → auto-send, else → human review queue
- Retrain weekly (cron job)

### Phase 6 — Dashboard & Cost Tracking
- Per-reply card: model, tokens in/out, cost, confidence score, expandable context
- Aggregate charts: daily cost, intent distribution, auto-send rate
- Threshold control with live preview of what would have been auto-sent

### Phase 7 — Deployment
- Containerise (Docker)
- Deploy alongside LumenX on Railway or separate service
- Configure polling interval (default 2.5 s, match platform)
- Secrets via env vars

---

## Project File Layout (target)

```
Ramco-Training/
├── CLAUDE.md
├── README.md
├── agent/
│   ├── main.py             # FastAPI app + polling loop
│   ├── intent_router.py
│   ├── context_builder.py
│   ├── llm_draft.py
│   ├── confidence_net.py
│   ├── feedback_log.py
│   └── lumenx_api.py       # thin wrapper around admin API
├── wiki/
│   ├── build_wiki.py       # fetch products → chunk → embed
│   ├── wiki_store.py       # FAISS/cosine retrieval
│   └── wiki_data/          # cached embeddings + chunks
├── dashboard/
│   ├── app.py              # Streamlit or FastAPI+Jinja2
│   └── templates/
├── models/
│   └── confidence_mlp.pkl  # trained MLP
├── data/
│   └── feedback_log.jsonl  # append-only feedback log
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## Environment Variables

```
ANTHROPIC_API_KEY=sk-ant-...
LUMENX_BASE_URL=https://lumenx-demo.up.railway.app
LUMENX_ADMIN_TOKEN=lmx_GQlch0Q5NOwVuVSADXRuFNJvxIpzVGwI
CONFIDENCE_THRESHOLD=0.80      # default, overridable from dashboard
POLL_INTERVAL_SECONDS=5
```

---

## Key Constraints Summary

- Never hallucinate pricing or refund policy
- Track every API call cost (input_tokens × price + output_tokens × price)
- Haiku for cheap tasks, Sonnet for reply generation
- All auto-sends require confidence ≥ threshold
- Human review queue is always available as fallback
- Feedback log is append-only (audit trail)
