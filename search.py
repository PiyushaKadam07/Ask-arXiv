"""Find the arXiv papers that best match a question.

Two kinds of search, combined into one:
  keyword (BM25)        good at exact terms: model names, acronyms, rare words
  meaning (embeddings)  finds papers that say the same thing in different words
"""
import json
import re
from pathlib import Path
from typing import Callable

import numpy as np
from rank_bm25 import BM25Okapi

DATA = Path(__file__).parent / "data"
PAPERS_FILE = DATA / "arxiv_sample.jsonl"
EMBEDDINGS_FILE = DATA / "embeddings.npy"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models find better matches when the question (not the papers) gets this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

STOPWORDS = set("""a an and are as at be by can do does for from has have how in is it its of on or
that the their this to was we what when which while who why will with""".split())

Embedder = Callable[[list[str]], np.ndarray]


def tokenize(text: str) -> list[str]:
    """Lowercase words; keeps terms like "b-tree" and "gpt-4" whole and drops filler words."""
    words = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())
    return [w for w in words if w not in STOPWORDS]


def paper_text(paper: dict) -> str:
    return f"{paper['title']}. {paper['abstract']}"


def load_papers(path: Path = PAPERS_FILE) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def fastembed_model() -> Embedder:
    """The real embedding model. fastembed runs it with ONNX, so we don't need PyTorch."""
    from fastembed import TextEmbedding

    model = TextEmbedding(EMBED_MODEL)
    return lambda texts: np.array(list(model.embed(texts, batch_size=64)), dtype=np.float32)


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector to length 1, so a dot product is the cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


class Searcher:
    def __init__(self, papers: list[dict], embeddings: np.ndarray, embed: Embedder):
        self.papers = papers
        self.embeddings = normalize(embeddings.astype(np.float32))
        self.embed = embed
        self.bm25 = BM25Okapi([tokenize(paper_text(p)) for p in papers])

    @classmethod
    def load(cls) -> "Searcher":
        return cls(load_papers(), np.load(EMBEDDINGS_FILE), fastembed_model())

    def keyword(self, question: str, k: int = 5) -> list[int]:
        """Indexes of the top k papers by BM25 score."""
        scores = self.bm25.get_scores(tokenize(question))
        return [int(i) for i in np.argsort(-scores)[:k] if scores[i] > 0]

    def meaning(self, question: str, k: int = 5) -> list[int]:
        """Indexes of the top k papers whose embeddings are closest to the question's."""
        query = normalize(self.embed([QUERY_PREFIX + question])[0])
        scores = self.embeddings @ query
        return [int(i) for i in np.argsort(-scores)[:k]]

    def hybrid(self, question: str, k: int = 5, pool: int = 50) -> list[int]:
        """Combine both rankings with reciprocal rank fusion (RRF).

        Each paper gets 1 / (60 + rank) from each list it appears in. A paper near
        the top of both lists wins. RRF only uses ranks, so we don't have to make
        BM25 scores and cosine similarities comparable.
        """
        fused: dict[int, float] = {}
        for ranking in (self.keyword(question, pool), self.meaning(question, pool)):
            for rank, index in enumerate(ranking, start=1):
                fused[index] = fused.get(index, 0.0) + 1.0 / (60 + rank)
        return sorted(fused, key=fused.get, reverse=True)[:k]

    def search(self, question: str, k: int = 5, mode: str = "hybrid") -> list[dict]:
        method = {"keyword": self.keyword, "meaning": self.meaning, "hybrid": self.hybrid}[mode]
        return [self.papers[i] for i in method(question, k)]


if __name__ == "__main__":
    import sys

    searcher = Searcher.load()
    question = " ".join(sys.argv[1:]) or "how do graph neural networks handle over-smoothing?"
    for mode in ("keyword", "meaning", "hybrid"):
        print(f"\n{mode}:")
        for paper in searcher.search(question, mode=mode):
            print(f"  [{paper['id']}] {paper['title']}")