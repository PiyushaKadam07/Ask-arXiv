"""Turn every paper into an embedding once and save them, so the app starts fast.

  python build_index.py     # about 10-20 minutes on a 2-core Codespace
"""
import time

import numpy as np

from search import EMBEDDINGS_FILE, fastembed_model, load_papers, paper_text

papers = load_papers()
embed = fastembed_model()
texts = [paper_text(p) for p in papers]

start = time.perf_counter()
batches = []
for i in range(0, len(texts), 500):
    batches.append(embed(texts[i:i + 500]))
    done = min(i + 500, len(texts))
    print(f"{done:,} / {len(texts):,} papers  ({time.perf_counter() - start:.0f} s)", flush=True)

# float16 halves the file size (about 15 MB) and doesn't change the rankings in practice
np.save(EMBEDDINGS_FILE, np.concatenate(batches).astype(np.float16))
print(f"Saved {EMBEDDINGS_FILE}")