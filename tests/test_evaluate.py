from evaluate import answer_scores, search_scores
from test_search import searcher  # the tiny 4-paper searcher from the search tests

QUESTIONS = [
    {"paper_id": "2", "question": "how to protect training data with noise"},  # found 1st
    {"paper_id": "3", "question": "robot"},  # found 1st
    {"paper_id": "4", "question": "zzz nothing matches"},  # not found
]


def test_recall_and_mrr_count_the_right_paper():
    row = search_scores(searcher, QUESTIONS, "keyword")
    assert row["recall@5"] == 66.7  # 2 of the 3 papers are in the top 5
    assert row["mrr@10"] == 0.667  # (1/1 + 1/1 + 0) / 3


def test_answer_scores_only_judge_citing_when_the_paper_was_found():
    def generate(prompt):
        if "Question: robot" in prompt:
            return "Robots use cameras [1].", "fake"
        return "The papers I found don't answer this.", "fake"

    row = answer_scores(searcher, generate, QUESTIONS[1:2], ["cake?", "football?"], pause=0)
    assert row["cited_right_paper"] == 100.0
    assert row["off_topic_refused"] == 100.0