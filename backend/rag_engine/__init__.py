# ── RAG Engine Package ────────────────────────────────────────────────────────
#  What this package does:
#  It does 3 things:
#    1. Checks if the user has permission to access data (RBAC)
#    2. Finds relevant documents from ChromaDB (Retrieval)
#    3. Sends those documents + question to LLaMA to get an answer (Generation)
#  Together this is called RAG — Retrieval Augmented Generation
#
#  This __init__.py re-exports ask_question so that existing imports like
#  `from backend.rag_engine import ask_question` continue to work unchanged.
# ──────────────────────────────────────────────────────────────────────────────

from .pipeline import ask_question
from .rbac import enforce_rbac, ROLE_FOLDERS, VALID_ROLES

__all__ = ["ask_question", "enforce_rbac", "ROLE_FOLDERS", "VALID_ROLES"]
