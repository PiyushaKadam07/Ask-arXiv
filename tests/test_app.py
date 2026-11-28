from fastapi.testclient import TestClient

from app import RateLimiter, create_app
from test_search import searcher  # the tiny 4-paper searcher


def fake_llm(prompt):
    if "robot" in prompt.lower().split("question:")[-1]:
        return "A camera guides the gripper [1].", "fake-model"
    return "The papers I found don't answer this; they cover graphs [1].", "fake-model"


def client():
    return TestClient(create_app(searcher, fake_llm))


def test_page_and_health():
    with client() as c:
        assert "Ask arXiv" in c.get("/").text
        assert c.get("/health").json() == {"status": "ok", "papers": 4}


def test_ask_returns_answer_sources_and_refused_flag():
    with client() as c:
        body = c.post("/ask", json={"question": "how do robot grippers work?"}).json()
        assert body["answer"].endswith("[1].")
        assert body["refused"] is False
        assert body["sources"][0]["url"].startswith("https://arxiv.org/abs/")

        off_topic = c.post("/ask", json={"question": "best chocolate cake?"}).json()
        assert off_topic["refused"] is True


def test_rejects_empty_and_very_long_questions():
    with client() as c:
        assert c.post("/ask", json={"question": ""}).status_code == 422
        assert c.post("/ask", json={"question": "x" * 301}).status_code == 422


def test_busy_llm_gives_503_with_a_message():
    def busy(prompt):
        raise RuntimeError("Every Gemini model is busy right now.")

    with TestClient(create_app(searcher, busy)) as c:
        response = c.post("/ask", json={"question": "robot arms"})
        assert response.status_code == 503
        assert "busy" in response.json()["detail"]


def test_rate_limiter_allows_ten_a_minute_per_visitor():
    limiter = RateLimiter(limit=10, window=60)
    assert all(limiter.allow("a", now=t) for t in range(10))
    assert not limiter.allow("a", now=30)  # 11th within the minute
    assert limiter.allow("b", now=30)  # other visitors aren't affected
    assert limiter.allow("a", now=70)  # a minute later it's fine again