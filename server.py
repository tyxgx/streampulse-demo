"""StreamPulse RAG demo — FastAPI app (single process, ~350 MB RAM).

Pipeline (same ideas as the production Django app's apps/chatbot/rag.py):
  small-talk shortcut -> SQL router (exact aggregates) -> embed -> nearest chunks
  -> confidence gate (refuse instead of hallucinate) -> grounded LLM answer with sources.
"""
import re
import threading
from contextlib import asynccontextmanager
import time
from collections import OrderedDict, defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import retrieval
import sql_layer
from llm import AllProvidersUnavailable, SMALLTALK_SYSTEM_PROMPT, SYSTEM_PROMPT, call_llm

@asynccontextmanager
async def lifespan(_):
    # Warm everything so the first visitor doesn't pay for model load / page-in.
    retrieval.search_chunks(retrieval.embed_query("warm up"), top_k=1)
    sql_layer.find_country("warm up")
    yield


app = FastAPI(title="StreamPulse RAG demo", lifespan=lifespan)
STATIC = Path(__file__).parent / "static"

_GREETINGS = {"hi", "hello", "hey", "yo", "sup"}
_THANKS = ("thanks", "thank you", "thx", "ty", "cheers")
_META = ("who are you", "what are you", "what can you do", "what do you do", "help me",
         "how does this work", "what can i ask")

_cache: "OrderedDict[str, dict]" = OrderedDict()
_cache_lock = threading.Lock()
_hits = defaultdict(deque)
RATE_PER_MIN = 15


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=400)


def is_smalltalk(q: str) -> bool:
    low = q.strip().lower().rstrip("!.?")
    words = low.split()
    if len(words) <= 4 and (low in _GREETINGS or (words and words[0] in _GREETINGS)
                            or any(low.startswith(t) for t in _THANKS)):
        return True
    return any(p in low for p in _META)


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "?")


def answer(question: str) -> dict:
    if is_smalltalk(question):
        try:
            return {"answer": call_llm(question, system_prompt=SMALLTALK_SYSTEM_PROMPT), "route": "chat", "sources": []}
        except AllProvidersUnavailable as e:
            return {"answer": f"⚠️ {e}", "route": "error", "sources": []}

    intent = sql_layer.detect_sql_intent(question)
    if intent:
        result, desc = sql_layer.run_sql_intent(intent)
        if intent["kind"] == "count":
            text = f"There are **{result:,}** distinct {desc} in the data."
        elif intent["kind"] == "sum_streams":
            text = f"Total streams for **{desc}**: **{sql_layer.format_streams(result)}**."
        else:
            lines = [f"Here's the {desc}:", ""]
            lines += [f"{i}. **{n}** — {sql_layer.format_streams(t)}" for i, (n, t) in enumerate(result, 1)]
            text = "\n".join(lines)
        return {"answer": text, "route": "sql", "sources": []}

    q = retrieval.embed_query(question)
    country = sql_layer.find_country(question.lower())
    if country:
        chunks = retrieval.search_chunks_for_entity(q, "country_performance", country, top_k=5)
    else:
        chunks = retrieval.search_chunks(q, top_k=5)
        d = retrieval.top1_distance(chunks)
        if d is None or d > retrieval.NO_MATCH_DISTANCE_THRESHOLD:
            return {"answer": retrieval.NO_DATA_REPLY, "route": "gate", "sources": []}

    context = "\n".join(f"- {c['chunk_text']}" for c in chunks)
    try:
        reply = call_llm(f"Context:\n{context}\n\nQuestion: {question}", system_prompt=SYSTEM_PROMPT)
    except AllProvidersUnavailable as e:
        return {"answer": f"⚠️ {e}", "route": "error", "sources": []}
    sources = [{"table": c["source_table"], "key": re.sub(r"^spotify:artist:[A-Za-z0-9]+", "artist", c["source_key"]),
                "distance": round(c["distance"], 3)} for c in chunks]
    return {"answer": reply, "route": "semantic", "sources": sources}


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(retrieval._matrix)}


@app.post("/api/ask")
def ask(body: Ask, request: Request):
    question = " ".join(body.question.split())
    key = question.lower()
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return {**_cache[key], "cached": True}

    now = time.monotonic()
    w = _hits[_client_ip(request)]
    while w and now - w[0] > 60:
        w.popleft()
    if len(w) >= RATE_PER_MIN:
        return JSONResponse({"detail": "Too many questions — try again in a minute."}, status_code=429)
    w.append(now)

    t0 = time.perf_counter()
    try:
        result = answer(question)
    except Exception as exc:  # never leak internals to the UI
        raise HTTPException(status_code=500, detail="Something went wrong answering that.") from exc
    result["ms"] = int((time.perf_counter() - t0) * 1000)
    if result["route"] != "error":
        with _cache_lock:
            _cache[key] = result
            while len(_cache) > 200:
                _cache.popitem(last=False)
    return {**result, "cached": False}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")
