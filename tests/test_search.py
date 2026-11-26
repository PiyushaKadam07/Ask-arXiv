import numpy as np

from search import Searcher, tokenize

PAPERS = [
    {"id": "1", "title": "Over-smoothing in graph neural networks", "abstract": "Deep GNN layers make node features too similar."},
    {"id": "2", "title": "Differential privacy for SGD", "abstract": "We add noise to gradients to protect training data."},
    {"id": "3", "title": "Robot grasping with vision", "abstract": "A camera guides the gripper to pick up objects."},
    {"id": "4", "title": "Message passing on graphs", "abstract": "Nodes exchange information with neighbours in each layer."},
]

# Pretend embeddings: each paper "means" one topic. Paper 4 is about graphs but never says "neural".
TOPICS = {"graph": 0, "privacy": 1, "robot": 2}
EMBEDDINGS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0.9, 0, 0.1]], dtype=np.float32)


def fake_embed(texts):
    """Puts a question on the axis of whichever topic word it mentions."""
    out = np.zeros((len(texts), 3), dtype=np.float32)
    for row, text in enumerate(texts):
        for word, axis in TOPICS.items():
            if word in text.lower():
                out[row, axis] = 1
    return out + 1e-6


searcher = Searcher(PAPERS, EMBEDDINGS, fake_embed)


def test_tokenize_keeps_hyphenated_terms_and_drops_filler():
    assert tokenize("How does the B-tree handle GPT-4 queries?") == ["b-tree", "handle", "gpt-4", "queries"]


def test_keyword_search_matches_exact_words():
    assert searcher.keyword("differential privacy", 1) == [1]


def test_keyword_search_ignores_papers_without_any_query_word():
    assert searcher.keyword("gripper", 5) == [2]


def test_meaning_search_finds_related_papers_without_shared_words():
    # Paper 4 never says "learning" or "neural", but it's a graph paper, so meaning search finds it
    assert set(searcher.meaning("graph learning", 2)) == {0, 3}


def test_hybrid_puts_papers_found_by_both_searches_first():
    top = searcher.hybrid("over-smoothing graph", 2)
    assert top[0] == 0  # first in keyword search and in meaning search


def test_search_returns_paper_records():
    assert searcher.search("robot", 1, mode="meaning")[0]["id"] == "3"