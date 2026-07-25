# ── Embedding Model Loader ────────────────────────────────────────────────────
# This module loads the HuggingFace embedding model once (singleton pattern)
# and reuses it for all queries, avoiding reloading the 90MB model every time.
# ──────────────────────────────────────────────────────────────────────────────

# ── Step 3: Load embedding model once ─────────────────────────────────────────
# The embedding model converts text into vectors (lists of numbers).
# Similar text produces similar vectors — this is what powers semantic search.
# We load it once and reuse it for all queries (singleton pattern).
# Without this, the 90MB model would reload on every single query — very slow.

_embedding_model = None  # starts as None, gets filled on first use

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        # Lazy load to prevent Render boot timeout
        from langchain_community.embeddings import HuggingFaceEmbeddings
        # Load the model from HuggingFace (only happens once)
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model
