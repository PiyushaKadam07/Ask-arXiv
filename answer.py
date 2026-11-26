"""Answer a question about arXiv papers with Gemini, citing the papers it used.

  python answer.py "how do graph neural networks handle over-smoothing?"
"""
import os
import re
import time
from typing import Callable

from google import genai
from google.genai import errors, types

# Tried in order: if one is busy (503) or out of free quota (429), move to the next.
MODELS = os.getenv("GEMINI_MODELS", "gemini-3.5-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite").split(",")
RETRYABLE = {429, 500, 502, 503, 504}

INSTRUCTIONS = """You answer questions about computer science research using only the numbered
paper abstracts you are given.

- Base every statement on the abstracts. Don't add facts from memory.
- Cite the papers you use with their numbers in square brackets, like [1] or [2, 4], right after
  the sentence they support.
- If the abstracts don't answer the question, say "The papers I found don't answer this." and,
  if it helps, say in one sentence what they do cover.
- Keep it short: one or two paragraphs, plain language, no headings."""

NO_ANSWER = "The papers I found don't answer this."

Generate = Callable[[str], tuple[str, str]]  # prompt -> (answer text, model used)


def build_prompt(question: str, papers: list[dict]) -> str:
    sources = "\n\n".join(
        f"[{n}] {p['title']} ({p['id']})\n{p['abstract']}" for n, p in enumerate(papers, start=1)
    )
    return f"Abstracts:\n\n{sources}\n\nQuestion: {question}"


def gemini(client: genai.Client | None = None, models: list[str] = MODELS, tries: int = 2) -> Generate:
    """Calls Gemini, retrying busy models with a short wait and then falling back to the next one."""
    client = client or genai.Client()  # reads GEMINI_API_KEY
    config = types.GenerateContentConfig(system_instruction=INSTRUCTIONS, temperature=0.2)

    def generate(prompt: str) -> tuple[str, str]:
        last_error: Exception | None = None
        for model in models:
            for attempt in range(tries):
                try:
                    reply = client.models.generate_content(model=model, contents=prompt, config=config)
                    return (reply.text or "").strip(), model
                except errors.APIError as err:
                    if err.code not in RETRYABLE:
                        raise  # e.g. a bad API key: retrying won't help
                    last_error = err
                    time.sleep(2**attempt)  # 1 s, then 2 s
        raise RuntimeError("Every Gemini model is busy right now. Please try again in a minute.") from last_error

    return generate


def cited_numbers(text: str) -> set[int]:
    """Every number cited as [1] or [2, 4]."""
    found = set()
    for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", text):
        found.update(int(n) for n in group.split(","))
    return found


def remove_bad_citations(text: str, count: int) -> str:
    """Drop citations to papers that don't exist (like [7] when there were 5), so every link works."""
    def fix(match: re.Match) -> str:
        valid = [n for n in re.split(r"\s*,\s*", match.group(1)) if 1 <= int(n) <= count]
        return f"[{', '.join(valid)}]" if valid else ""

    text = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", fix, text)
    return re.sub(r" +([.,;])", r"\1", text)  # tidy up a space left before punctuation


def answer(question: str, searcher, generate: Generate, k: int = 5) -> dict:
    papers = searcher.search(question, k=k)
    if not papers:
        return {"answer": NO_ANSWER, "sources": [], "model": None}

    text, model = generate(build_prompt(question, papers))
    text = remove_bad_citations(text, len(papers))
    cited = cited_numbers(text)
    sources = [
        {
            "number": n,
            "id": p["id"],
            "title": p["title"],
            "url": f"https://arxiv.org/abs/{p['id']}",
            "cited": n in cited,
        }
        for n, p in enumerate(papers, start=1)
    ]
    return {"answer": text, "sources": sources, "model": model}


if __name__ == "__main__":
    import sys

    from search import Searcher

    question = " ".join(sys.argv[1:]) or "how do graph neural networks handle over-smoothing?"
    result = answer(question, Searcher.load(), gemini())
    print(f"\n{result['answer']}\n\n(answered by {result['model']})\n")
    for s in result["sources"]:
        mark = "*" if s["cited"] else " "
        print(f" {mark}[{s['number']}] {s['title']}\n     {s['url']}")