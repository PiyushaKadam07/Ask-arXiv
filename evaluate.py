"""Measure how well Ask arXiv finds and uses the right papers.

1. Search: for each test question, is its paper in the top 5 (recall@5), and how high (MRR)?
   Compared for keyword, meaning and hybrid search. No Gemini calls.
2. Answers (with --answers): on some of the questions, does Gemini cite the right paper
   when the search found it? And does it refuse questions no CS paper can answer?

  python evaluate.py                 # search only, takes a minute
  python evaluate.py --answers 50    # also 50 answers + 20 off-topic questions (about 70 Gemini calls)
"""
import argparse
import json
import time

from answer import NO_ANSWER, answer, gemini
from make_questions import QUESTIONS_FILE
from search import DATA, Searcher

OFF_TOPIC = [
    "What is the best recipe for chocolate cake?",
    "Who won the 2018 FIFA World Cup?",
    "How do I get a refund for a cancelled flight?",
    "What are good exercises for lower back pain?",
    "How often should I water a cactus?",
    "What is the capital of Australia?",
    "How do I remove a red wine stain from a carpet?",
    "Which car has the best fuel economy this year?",
    "What should I pack for a week in Iceland?",
    "How long should I boil an egg for a runny yolk?",
    "Is it better to rent or buy a house?",
    "What are the rules of cricket?",
    "How do I train my puppy to stop biting?",
    "What is the plot of Pride and Prejudice?",
    "How much coffee is safe to drink in a day?",
    "What is a good beginner guitar?",
    "How do I change a flat tyre?",
    "Which wine goes well with salmon?",
    "When is the best time to visit Japan?",
    "How do I knit a scarf?",
]


def load_questions() -> list[dict]:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def search_scores(searcher: Searcher, questions: list[dict], mode: str, k: int = 5, depth: int = 10) -> dict:
    """recall@k: share of questions whose paper is in the top k.
    MRR@depth: average of 1/rank of the right paper (0 if it's not in the top `depth`)."""
    method = {"keyword": searcher.keyword, "meaning": searcher.meaning, "hybrid": searcher.hybrid}[mode]
    ids = [p["id"] for p in searcher.papers]
    hits, reciprocal = 0, 0.0
    for q in questions:
        ranked = [ids[i] for i in method(q["question"], depth)]
        if q["paper_id"] in ranked[:k]:
            hits += 1
        if q["paper_id"] in ranked:
            reciprocal += 1 / (ranked.index(q["paper_id"]) + 1)
    return {"mode": mode, "recall@5": round(100 * hits / len(questions), 1), "mrr@10": round(reciprocal / len(questions), 3)}


def answer_scores(searcher: Searcher, generate, questions: list[dict], off_topic: list[str], pause: float = 4.0) -> dict:
    found = cited = 0
    for q in questions:
        result = answer(q["question"], searcher, generate)
        source = next((s for s in result["sources"] if s["id"] == q["paper_id"]), None)
        if source:  # only judge citing when the search actually handed Gemini the paper
            found += 1
            cited += source["cited"]
        time.sleep(pause)

    refused = 0
    for question in off_topic:
        refused += answer(question, searcher, generate)["answer"].startswith(NO_ANSWER)
        time.sleep(pause)

    return {
        "answered": len(questions),
        "cited_right_paper": round(100 * cited / found, 1) if found else None,
        "off_topic_refused": round(100 * refused / len(off_topic), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", type=int, default=0, help="how many questions to answer with Gemini")
    args = parser.parse_args()

    searcher = Searcher.load()
    questions = load_questions()
    print(f"{len(questions)} test questions\n")

    results = {"questions": len(questions), "search": []}
    for mode in ("keyword", "meaning", "hybrid"):
        row = search_scores(searcher, questions, mode)
        results["search"].append(row)
        print(f"{mode:8}  recall@5 {row['recall@5']:5.1f}%   MRR@10 {row['mrr@10']:.3f}", flush=True)

    if args.answers:
        row = answer_scores(searcher, gemini(), questions[:args.answers], OFF_TOPIC)
        results["answers"] = row
        print(f"\nGemini cited the right paper in {row['cited_right_paper']}% of answers where search found it")
        print(f"Off-topic questions refused: {row['off_topic_refused']}%")

    with open(DATA / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {DATA / 'eval_results.json'}")


if __name__ == "__main__":
    main()