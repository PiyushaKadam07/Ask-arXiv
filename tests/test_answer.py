import pytest
from google.genai import errors

from answer import NO_ANSWER, answer, build_prompt, cited_numbers, gemini, remove_bad_citations

PAPERS = [
    {"id": "2501.00762", "title": "Residual connections mitigate oversmoothing", "abstract": "Skip connections keep node features apart."},
    {"id": "2605.00951", "title": "Graph rewiring survey", "abstract": "Rewiring edges reduces over-squashing and over-smoothing."},
]


class FakeSearcher:
    def __init__(self, papers):
        self.papers = papers

    def search(self, question, k=5):
        return self.papers[:k]


def test_prompt_numbers_each_abstract():
    prompt = build_prompt("why?", PAPERS)
    assert "[1] Residual connections mitigate oversmoothing (2501.00762)" in prompt
    assert "[2] Graph rewiring survey" in prompt
    assert prompt.endswith("Question: why?")


def test_finds_single_and_grouped_citations():
    assert cited_numbers("Skip connections help [1]. Rewiring too [1, 2].") == {1, 2}


def test_removes_citations_to_papers_that_were_not_given():
    assert remove_bad_citations("A [1]. B [7]. C [2, 9].", count=2) == "A [1]. B. C [2]."


def test_answer_marks_which_sources_were_cited():
    result = answer("over-smoothing?", FakeSearcher(PAPERS), lambda prompt: ("Use skip connections [1].", "fake-model"))
    assert result["model"] == "fake-model"
    assert [s["cited"] for s in result["sources"]] == [True, False]
    assert result["sources"][0]["url"] == "https://arxiv.org/abs/2501.00762"


def test_answer_without_matching_papers_does_not_call_the_llm():
    def must_not_run(prompt):
        raise AssertionError("the LLM should not be called")

    assert answer("zzz", FakeSearcher([]), must_not_run)["answer"] == NO_ANSWER


# --- retries and fallback, with a pretend Gemini client ---

class Reply:
    def __init__(self, text):
        self.text = text


class FakeClient:
    """Fails with the given error codes in order, then answers."""

    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        if self.failures:
            code = self.failures.pop(0)
            raise errors.APIError(code, {"error": {"code": code, "message": "nope", "status": "X"}})
        return Reply(f"answer from {model}")


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr("answer.time.sleep", lambda seconds: None)


def test_busy_model_is_retried_then_next_model_is_used():
    client = FakeClient([503, 503])
    text, model = gemini(client, models=["big", "small"], tries=2)("q")
    assert client.calls == ["big", "big", "small"]
    assert model == "small"


def test_bad_request_is_not_retried():
    client = FakeClient([400])
    with pytest.raises(errors.APIError):
        gemini(client, models=["big", "small"])("q")
    assert client.calls == ["big"]


def test_gives_a_clear_message_when_every_model_is_busy():
    client = FakeClient([429] * 4)
    with pytest.raises(RuntimeError, match="busy"):
        gemini(client, models=["big", "small"], tries=2)("q")