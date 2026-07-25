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
    # Lazy load heavy dependencies to prevent Render boot timeouts
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma

    embedding   = get_embedding_model()
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, ".."))
    persist_dir = os.path.join(DATA_DIR, f"chroma_db_{role}")  # e.g. chroma_db_finance, chroma_db_hr
    hash_file   = f"{persist_dir}/.hash" # where we save the fingerprint
    current_hash = compute_folder_hash(folders, base_path)

    needs_rebuild = True  # default: assume we need to rebuild

    # Check if DB already exists AND fingerprint matches current files
    if os.path.exists(persist_dir) and os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            saved_hash = f.read().strip()
        if saved_hash == current_hash:
            needs_rebuild = False  # files unchanged — safe to load existing DB

    if needs_rebuild:
        print(f"[INFO] Building ChromaDB for role: {role}")

        # Load all documents from allowed folders
        documents = load_documents(folders, base_path)

        if not documents:
            raise ValueError(f"No documents found for role '{role}'. Check your data folders.")

        # Split documents into chunks of 2000 characters with 200 overlap
        # Overlap ensures sentences at chunk boundaries aren't cut in half
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        ).split_documents(documents)

        # Delete old DB if it exists (removes stale vectors from deleted files)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        # Create new ChromaDB — converts chunks to vectors and saves to disk
        db = Chroma.from_documents(chunks, embedding, persist_directory=persist_dir)

        # Save fingerprint so next run can detect if files changed
        os.makedirs(persist_dir, exist_ok=True)
        with open(hash_file, "w") as f:
            f.write(current_hash)

    else:
        print(f"[INFO] Loading existing ChromaDB for role: {role}")
        # Load existing DB from disk — much faster than rebuilding
        db = Chroma(persist_directory=persist_dir, embedding_function=embedding)

    return db
