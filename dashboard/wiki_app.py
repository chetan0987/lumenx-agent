"""LumenX Wiki Explorer — FastAPI backend.

Usage (from project root):
    uvicorn dashboard.wiki_app:app --port 8000 --reload
"""

import json
import os
import sys
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

WIKI_DATA = ROOT / "wiki" / "wiki_data"
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="LumenX Wiki Explorer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _load_chunks() -> list[dict]:
    path = WIKI_DATA / "chunks.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="Wiki not built yet. Run: python -m wiki.build_wiki"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/stats")
async def stats():
    path = WIKI_DATA / "chunks.json"
    if not path.exists():
        return {"ready": False, "products": 0, "chunks": 0, "sections": []}
    with path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    products = len(set(c["product_id"] for c in chunks))
    sections = sorted(set(c["section"] for c in chunks))
    return {"ready": True, "products": products, "chunks": len(chunks), "sections": sections}


@app.get("/api/graph")
async def get_graph():
    chunks = _load_chunks()

    # ── Product nodes ──
    product_map: dict[str, dict] = {}
    for c in chunks:
        pid = c["product_id"]
        if pid not in product_map:
            product_map[pid] = {
                "id": f"p__{pid}",
                "product_id": pid,
                "label": c["product_name"],
                "type": "product",
                "r": 22,
                "sections": [],
            }
        product_map[pid]["sections"].append(c["section"])

    # ── Section/chunk nodes ──
    chunk_nodes = []
    for c in chunks:
        preview = c["text"]
        chunk_nodes.append({
            "id": c["id"],
            "product_id": c["product_id"],
            "product_name": c["product_name"],
            "label": c["section"].replace("_", " "),
            "type": "section",
            "section": c["section"],
            "r": 9,
            "text_preview": (preview[:160] + "…") if len(preview) > 160 else preview,
        })

    nodes = list(product_map.values()) + chunk_nodes

    # ── Edges: product → section ──
    edges = []
    for c in chunks:
        edges.append({
            "id": f"e__{c['id']}",
            "source": f"p__{c['product_id']}",
            "target": c["id"],
            "type": "has_section",
        })

    # ── Integration cross-edges: product ↔ product ──
    pname_to_id = {v["label"].lower(): v["id"] for v in product_map.values()}
    seen_cross: set[tuple] = set()
    for c in chunks:
        if c["section"] != "integrations":
            continue
        src = f"p__{c['product_id']}"
        text_l = c["text"].lower()
        for pname, pid in pname_to_id.items():
            if pid == src:
                continue
            if pname in text_l:
                key = tuple(sorted([src, pid]))
                if key not in seen_cross:
                    edges.append({
                        "id": f"cross__{src}__{pid}",
                        "source": src,
                        "target": pid,
                        "type": "integration",
                    })
                    seen_cross.add(key)

    return {"nodes": nodes, "edges": edges}


@app.get("/api/chunk/{chunk_id:path}")
async def get_chunk(chunk_id: str):
    chunks = _load_chunks()
    for c in chunks:
        if c["id"] == chunk_id:
            return c
    raise HTTPException(status_code=404, detail="Chunk not found")


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def query_wiki(req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        from wiki.wiki_store import search, is_ready
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"Wiki store unavailable: {exc}")

    if not is_ready():
        raise HTTPException(
            status_code=503,
            detail="Wiki not built yet. Run: python -m wiki.build_wiki"
        )

    hits = search(question, top_k=5)

    kb = "\n\n".join(
        f"[Source {i}: {h['product_name']} / {h['section']}]\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )

    prompt = (
        "You are a precise assistant that answers questions about LumenX SaaS products "
        "using ONLY the knowledge base excerpts provided below.\n\n"
        "Rules:\n"
        "- Base every claim on a specific source excerpt.\n"
        "- At the end of your answer, list which [Source N] references you used.\n"
        "- If the knowledge base does not contain enough information, say so clearly.\n"
        "- Be concise but complete.\n\n"
        f"KNOWLEDGE BASE:\n{kb}\n\n"
        f"QUESTION: {question}"
    )

    from agent.cost_logger import compute_cost, log_call

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="ANTHROPIC_API_KEY not set. Add it to your .env file."
        )
    client = anthropic.Anthropic(api_key=api_key)
    model = "claude-haiku-4-5-20251001"
    try:
        response = client.messages.create(
            model=model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
    except anthropic.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="Invalid Anthropic API key. Update ANTHROPIC_API_KEY in your .env file."
        )
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Anthropic API unreachable: {e}")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Anthropic API error: {e.message}")

    answer = response.content[0].text.strip()
    cost = compute_cost(model, response.usage.input_tokens, response.usage.output_tokens)
    log_call(
        "_wiki_query", model,
        response.usage.input_tokens, response.usage.output_tokens, cost,
        {"action": "wiki_query", "question": question[:120]},
    )

    return {
        "answer": answer,
        "sources": hits,
        "model": model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "cost_usd": cost,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("dashboard.wiki_app:app", host="0.0.0.0", port=8000, reload=True)
