# LumenX Auto-Reply Agent

An intelligent, human-in-the-loop customer support agent for the LumenX SaaS platform. The agent reads incoming customer messages, assembles rich context, drafts a reply with Claude, scores its own confidence with a tiny neural network, and either sends automatically or routes to human review.

---

## What It Does

```
Customer message
       │
       ▼
  Intent Router ──────── greeting/small-talk? → polite reply (no KB lookup)
       │
       ▼ (product query)
  Context Builder
   ├── Products JSON (live from LumenX API)
   ├── LLM Wiki (structured product knowledge, retrieved by similarity)
   ├── Current thread history
   ├── Summary of all past customer threads
   └── Feedback log (similar Q&A pairs from past replies)
       │
       ▼
  LLM Draft  (Claude Sonnet — quality reply)
       │
       ▼
  Confidence Net  (tiny MLP, trained on edit history)
   ├── score ≥ threshold → auto-send
   └── score < threshold → human review queue
                               │
                         edit / approve
                               │
                         feedback log updated → retrains MLP
```

---

## Architecture Details

### 1. Intent Router
- Model: `claude-haiku-4-5-20251001` (low cost)
- Categories: `greeting`, `pricing`, `technical`, `other`
- Greetings and small-talk are handled directly without querying the KB

### 2. Context Builder
- **Products JSON**: fetched from `/api/admin/products`, cached and refreshed nightly
- **LLM Wiki**: all 20 LumenX products chunked and embedded (Karpathy-style knowledge base). Top-K chunks retrieved per query.
- **Thread history**: full current conversation injected
- **Past thread summary**: nightly batch summarisation of all historical threads
- **Feedback log**: top similar past Q&A pairs injected as few-shot examples

### 3. LLM Draft
- Model: `claude-sonnet-4-6`
- System prompt enforces: no hallucination on pricing/refunds, professional + empathetic tone, cite sources
- Logs: `input_tokens`, `output_tokens`, `cost_usd` per reply

### 4. Confidence Net (MLP)
- Input features:
  - `edit_ratio` — how much past drafts were edited (training signal)
  - `intent_type` — one-hot encoded
  - `kb_hits` — number of KB chunks retrieved
  - `draft_len_tokens` — length of draft
  - `thread_turn` — conversation depth
  - `feedback_match_score` — similarity to best feedback log match
- Architecture: 2 hidden layers × 64 units, ReLU activations, sigmoid output
- Loss: binary cross entropy
- Label: `1` = sent as-is, `0` = human edited
- Training data collected organically during Phases 1–4

---

## Build Phases

| Phase | What Gets Built | Output |
|---|---|---|
| 1 | API integration + basic agent + cost logging | Working replies, all human-reviewed |
| 2 | LLM Wiki + Context Builder | Replies use full product knowledge |
| 3 | Intent Router | Cheap, fast routing; greetings handled cleanly |
| 4 | Human Review UI + Feedback Loop | Web dashboard, edit/approve flow, feedback log |
| 5 | Confidence Net | MLP trained on collected data; auto-send live |
| 6 | Dashboard & Cost Tracking | Per-reply cost, tokens, expandable context |
| 7 | Deployment | Dockerised, deployed on Railway |

---

## Getting Ground Truth for the MLP

A chicken-and-egg problem: the MLP needs labeled training data, but you need the system running to collect it.

**Solution — organic data collection during Phases 1–4:**
- Every reply passes through human review (threshold = 1.0, nothing auto-sends)
- Approved without edit → label `1`
- Edited before sending → label `0`, edit_ratio stored
- After ~200 examples: first MLP training run
- After ~500 examples: threshold slider activated in dashboard
- System continuously retrains weekly as more labeled data accumulates

This means the first ~2–4 weeks of operation are the training data collection period. The MLP gets smarter the more the agent is used.

---

## Dashboard Features

- **Inbox view**: pending threads, draft preview, edit box, confidence score badge
- **Per-reply detail**: model used, tokens in/out, cost USD, confidence score, expandable full context window
- **Aggregate stats**: daily cost chart, intent distribution pie, auto-send rate over time
- **Threshold control**: slider (0.5–0.99) with live preview of historical sends at that threshold

---

## Guardrails

- **No hallucination on pricing or refunds.** If the agent lacks the specific data, it says: *"I don't have that specific information right now — our team will follow up shortly."*
- **Pricing and refund fields are always sourced from the live API**, never inferred.
- Human review is always one click away regardless of confidence score.
- Feedback log is append-only (full audit trail).

---

## Project Structure

```
Ramco-Training/
├── CLAUDE.md                  # Claude Code guidance (this project's AI instructions)
├── README.md                  # This file
├── agent/
│   ├── main.py                # FastAPI app + polling loop
│   ├── intent_router.py       # Haiku-based intent classification
│   ├── context_builder.py     # Assembles context from all sources
│   ├── llm_draft.py           # Sonnet reply generation
│   ├── confidence_net.py      # MLP training + inference
│   ├── feedback_log.py        # JSONL append + similarity retrieval
│   └── lumenx_api.py          # Admin API wrapper
├── wiki/
│   ├── build_wiki.py          # Fetch products → chunk → embed
│   ├── wiki_store.py          # Retrieval (cosine / FAISS)
│   └── wiki_data/             # Cached embeddings + chunks
├── dashboard/
│   ├── app.py                 # Streamlit or FastAPI+Jinja2 UI
│   └── templates/
├── models/
│   └── confidence_mlp.pkl     # Trained MLP
├── data/
│   └── feedback_log.jsonl     # Append-only feedback log
├── requirements.txt
├── .env.example
└── Dockerfile
```

---

## Environment Variables

```bash
ANTHROPIC_API_KEY=sk-ant-...
LUMENX_BASE_URL=https://lumenx-demo.up.railway.app
LUMENX_ADMIN_TOKEN=lmx_GQlch0Q5NOwVuVSADXRuFNJvxIpzVGwI
CONFIDENCE_THRESHOLD=0.80
POLL_INTERVAL_SECONDS=5
```

---

## Technology Stack

| Component | Choice | Reason |
|---|---|---|
| Backend | FastAPI (Python) | Async polling, easy API wrappers |
| LLM | Anthropic Claude (Haiku + Sonnet) | Cost-aware model selection |
| Embeddings | `sentence-transformers` or Claude embed | Local, no extra API cost |
| Vector search | FAISS or cosine (numpy) | Simple, no extra infra |
| Confidence Net | scikit-learn MLPClassifier or PyTorch | Fast to train on small data |
| Dashboard | Streamlit | Fast to build, good charts |
| Deployment | Docker + Railway | Matches LumenX platform |

---

## LumenX Platform

- **20 products** (e.g. EmailPilot, InvoiceFlow, TaskGrid)
- Each product has: pricing tiers, features, refund policy, cancellation terms, integration list, target audience, support SLA
- Company-wide policies: refund window, free trial, discounts
- Full dump available at `GET /api/admin/export`
