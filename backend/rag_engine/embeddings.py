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
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is missing! Please add it to Render.")
        
        from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
        # Hosted API embeddings: 0MB local memory footprint
        _embedding_model = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token,
            model_name="BAAI/bge-small-en-v1.5"
        )
    return _embedding_model
