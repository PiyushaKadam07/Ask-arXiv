"""Build the test set: one question per paper, for 200 random papers.

Gemini reads each abstract and writes a question that the paper answers, the way a
researcher would ask it. Then we know the "right" paper for every question.

  python make_questions.py      # about 20 Gemini calls, 10 papers per call
"""
import json
import random
import time

from google import genai
from google.genai import types
from pydantic import BaseModel

from answer import MODELS
from search import DATA, load_papers

QUESTIONS_FILE = DATA / "eval_questions.jsonl"
COUNT, BATCH, SEED = 200, 10, 7

INSTRUCTIONS = """For each paper abstract, write one question that a researcher might type into a
search box and that this paper answers.

- Ask about the problem or the finding, not about "this paper".
- Don't copy distinctive phrases from the title, and don't use names the authors invented
  (like their model's name). A real user wouldn't know them yet.
- One sentence, under 25 words."""


class Question(BaseModel):
    paper_id: str
    question: str


def write_questions(client: genai.Client, papers: list[dict]) -> list[Question]:
    abstracts = "\n\n".join(f"paper_id: {p['id']}\ntitle: {p['title']}\n{p['abstract']}" for p in papers)
    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS,
        temperature=0.7,
        response_mime_type="application/json",  # ask for JSON in this exact shape
        response_schema=list[Question],
    )
    for model in MODELS:
        try:
            reply = client.models.generate_content(model=model, contents=abstracts, config=config)
            return reply.parsed
        except Exception as err:  # busy or out of quota: try the next model
            print(f"  {model} failed ({err.__class__.__name__}), trying the next one")
            time.sleep(5)
    raise RuntimeError("Every model failed; wait a few minutes and run it again.")


def main():
    papers = random.Random(SEED).sample(load_papers(), COUNT)  # fixed seed: same 200 papers every time
    client = genai.Client()
    done = []
    for i in range(0, COUNT, BATCH):
        batch = papers[i:i + BATCH]
        wanted = {p["id"] for p in batch}
        done += [q for q in write_questions(client, batch) if q.paper_id in wanted]
        print(f"{len(done)} / {COUNT} questions", flush=True)
        time.sleep(4)  # stay under the free tier's requests-per-minute limit

    with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
        for q in done:
            f.write(json.dumps(q.model_dump()) + "\n")
    print(f"Saved {QUESTIONS_FILE}")


if __name__ == "__main__":
    main()