"""Web server for Ask arXiv: the question page, and the /ask endpoint behind it.

  uvicorn app:app --port 8000
"""
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from answer import NO_ANSWER, answer, gemini

PAGE = Path(__file__).parent / "static" / "index.html"
LIMIT, WINDOW = 10, 60  # questions per visitor per minute, to protect the free Gemini quota


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=300)


class RateLimiter:
    """Remembers when each visitor asked recently and turns them away after LIMIT per WINDOW."""

    def __init__(self, limit: int = LIMIT, window: float = WINDOW):
        self.limit, self.window = limit, window
        self.asked: dict[str, deque] = defaultdict(deque)

    def allow(self, visitor: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        times = self.asked[visitor]
        while times and now - times[0] > self.window:
            times.popleft()
        if len(times) >= self.limit:
            return False
        times.append(now)
        return True


def create_app(searcher=None, generate=None) -> FastAPI:
    """Tests pass in a small searcher and a fake LLM; the real app loads both at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from search import Searcher

        app.state.searcher = searcher or Searcher.load()  # about 5 s: papers, BM25, embedding model
        app.state.generate = generate or gemini()
        yield

    app = FastAPI(title="Ask arXiv", lifespan=lifespan)
    limiter = RateLimiter()

    @app.get("/", include_in_schema=False)
    def page():
        return FileResponse(PAGE)

    @app.get("/health")
    def health():
        return {"status": "ok", "papers": len(app.state.searcher.papers)}

    @app.post("/ask")
    def ask(body: Question, request: Request):
        # Cloud Run puts the visitor's real address first in X-Forwarded-For
        visitor = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
        if not limiter.allow(visitor):
            raise HTTPException(429, "You're asking very quickly. Please wait a minute and try again.")
        try:
            result = answer(body.question, app.state.searcher, app.state.generate)
        except RuntimeError as err:  # every Gemini model was busy
            raise HTTPException(503, str(err))
        result["refused"] = result["answer"].startswith(NO_ANSWER.rstrip("."))
        return result

    return app


app = create_app()