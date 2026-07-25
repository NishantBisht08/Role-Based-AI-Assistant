# ── Vector Store Manager ──────────────────────────────────────────────────────
# This module handles building or loading the ChromaDB vector database.
# Each role gets its own separate ChromaDB folder on disk.
# Smart logic detects if source files changed and rebuilds only when needed.
# ──────────────────────────────────────────────────────────────────────────────

import os
import shutil    # for deleting folders when we need to rebuild ChromaDB

from .embeddings import get_embedding_model
from .document_loader import compute_folder_hash, load_documents

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Step 7: Build or load ChromaDB ────────────────────────────────────────────
# ChromaDB stores all document chunks as vectors on disk.
# Each role gets its own separate ChromaDB folder.
# Smart logic:
#   - If data files haven't changed → load existing DB (fast)
#   - If data files changed → wipe old DB and rebuild (accurate)

def get_or_build_vectorstore(role: str, folders: list, base_path: str):
    from langchain_pinecone import PineconeVectorStore
    
    embedding = get_embedding_model()
    pinecone_index = os.environ.get("PINECONE_INDEX_NAME", "company-docs")
    
    # We no longer read PDFs or chunk them here!
    # The production server just connects to the pre-seeded Pinecone cloud index.
    # We use the 'role' as the namespace so finance vectors don't mix with hr vectors.
    
    print(f"[INFO] Connecting to Pinecone Hosted Vector Database for role: {role}")
    db = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index,
        embedding=embedding,
        namespace=role
    )

    return db
