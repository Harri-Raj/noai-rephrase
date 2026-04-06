"""
NoAI Rephrase — FastAPI Backend (MVP edition, no Supabase auth required)
Run:  uvicorn main:app --reload --port 8000
Env:  GEMINI_API_KEY=your_key_here
"""

import os
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Import the pipeline ────────────────────────────────────────────────────────
# Place pipeline.py in the same directory as this file, OR in humanizer/pipeline.py
try:
    from humanizer.pipeline import humanize, compute_ai_score
except ModuleNotFoundError:
    from pipeline import humanize, compute_ai_score   # flat layout fallback

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s │ %(name)s │ %(message)s")
logger = logging.getLogger("noai")

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set.\n"
        "Get a free key at https://aistudio.google.com and run:\n"
        "  export GEMINI_API_KEY=your_key_here"
    )

FREE_WORD_LIMIT = 300  # words per request for unauthenticated users

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NoAI Rephrase API",
    version="1.0.0",
    description="Transform AI-generated text into natural human writing.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Open for local dev — lock this down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class HumanizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)
    mode: str = Field(default="standard", pattern="^(standard|aggressive|research)$")


class HumanizeResponse(BaseModel):
    original: str
    simplified: str          # intermediate step — shown in the UI
    humanized: str
    ai_score_before: float
    ai_score_after: float
    similarity_score: float
    changes_made: int
    success: bool
    mode: str
    message: Optional[str] = None


class ScoreRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10_000)


class ScoreResponse(BaseModel):
    score: float
    label: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "noai-rephrase"}


@app.post("/api/humanize", response_model=HumanizeResponse)
async def api_humanize(payload: HumanizeRequest, request: Request):
    word_count = len(payload.text.split())

    if word_count > FREE_WORD_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"Free limit is {FREE_WORD_LIMIT} words per request. "
                   f"Your text has {word_count} words — please trim it.",
        )

    try:
        result = humanize(payload.text, payload.mode, GEMINI_API_KEY)
    except Exception as exc:
        logger.error("Pipeline error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Humanization failed: {exc}")

    return HumanizeResponse(
        original=result.original,
        simplified=result.simplified,
        humanized=result.humanized,
        ai_score_before=result.ai_score_before,
        ai_score_after=result.ai_score_after,
        similarity_score=result.similarity_score,
        changes_made=result.changes_made,
        success=result.success,
        mode=result.mode,
        message=result.error,
    )


@app.post("/api/score", response_model=ScoreResponse)
async def api_score(payload: ScoreRequest):
    """Score text for AI-likeness. No auth required."""
    try:
        score = compute_ai_score(payload.text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if score >= 70:
        label = "Likely AI-generated"
    elif score >= 40:
        label = "Mixed — some AI patterns"
    else:
        label = "Likely human-written"

    return ScoreResponse(score=score, label=label)


# ── Global error handler ──────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
