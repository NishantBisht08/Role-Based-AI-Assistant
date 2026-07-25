# ── Embedding Model Loader ────────────────────────────────────────────────────
# This module loads the HuggingFace embedding model once (singleton pattern)
# and reuses it for all queries, avoiding reloading the 90MB model every time.
# ──────────────────────────────────────────────────────────────────────────────

# ── Step 3: Load embedding model once ─────────────────────────────────────────
# The embedding model converts text into vectors (lists of numbers).
# Similar text produces similar vectors — this is what powers semantic search.
# We load it once and reuse it for all queries (singleton pattern).
# Without this, the 90MB model would reload on every single query — very slow.

import os

_embedding_model = None  # starts as None, gets filled on first use

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
        _embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)
    return _embedding_model
