# ── Document Loader ───────────────────────────────────────────────────────────
# This module handles:
#   - Detecting if data files have changed (fingerprint hashing)
#   - Extracting quarter/year metadata from filenames
#   - Loading documents from allowed folders with metadata headers
# ──────────────────────────────────────────────────────────────────────────────

import os        # for reading files and folders
import hashlib   # for creating a fingerprint of files to detect changes

# LangChain loaders — each one knows how to read a different file type
from langchain_community.document_loaders import (
    TextLoader,      # reads .md and .txt files
    CSVLoader,       # reads .csv files (like hr_data.csv)
    PyPDFLoader,     # reads .pdf files
    Docx2txtLoader,  # reads .docx Word files
)


# ── Step 4: Detect if data files have changed ─────────────────────────────────
# We create a fingerprint (hash) of all files in the role's folders.
# If any file is added, edited, or deleted — the hash changes.
# This tells us to rebuild ChromaDB instead of using the old stale one.

def compute_folder_hash(folders: list, base_path: str) -> str:
    hasher = hashlib.md5()  # MD5 creates a 32-character fingerprint string

    for folder in sorted(folders):  # sorted = consistent order every time
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path)):  # sorted = consistent order
            fpath = os.path.join(folder_path, fname)
            stat = os.stat(fpath)  # get file info without opening the file
            # Include filename + last modified time + file size in the hash
            hasher.update(f"{fname}{stat.st_mtime}{stat.st_size}".encode())

    return hasher.hexdigest()  # returns something like "a3f8c2d19e4b..."


# ── Step 5: Extract quarter and year from filename ────────────────────────────
# This function reads the filename and figures out which quarter/year it belongs to.
# Example: "marketing_report_q1_2024.md" → quarter="Q1", year="2024"
# This is important for the semantic collision fix — see Step 6 for why.

def extract_file_tags(fname: str) -> dict:
    
    # 1. Clean up the filename manually (No Regex needed for this!)
    name = fname.lower()
    name = name.replace(".md", "").replace(".csv", "").replace(".pdf", "").replace(".docx", "").replace(".txt", "")
    name = name.replace("_", " ").replace("-", " ")
    name = name.strip()

    # 2. Set default values just in case we don't find anything
    final_quarter = "annual"
    final_year = ""

    # 3. Check for the quarter (Q1, Q2, Q3, Q4)
    if "q1" in name:
        final_quarter = "Q1"
    elif "q2" in name:
        final_quarter = "Q2"
    elif "q3" in name:
        final_quarter = "Q3"
    elif "q4" in name:
        final_quarter = "Q4"

    # 4. Check for the year (Looking for 2020 through 2029)
    # split() turns "marketing report 2024" into ["marketing", "report", "2024"]
    words_in_name = name.split() 
    for word in words_in_name:
        if word.startswith("202") and len(word) == 4:
            final_year = word
            break # Stop looking once we find the year

    # 5. Return the clean, easy-to-read dictionary
    return {
        "quarter": final_quarter,
        "year": final_year,
        "doc_type": name
    }


# ── Step 6: Load documents from allowed folders ───────────────────────────────
# This function reads all files from the role's allowed folders.
# KEY FIX: We prepend a metadata header to every chunk BEFORE embedding.
#
# Why? Because Q1 and Q4 financial reports talk about the same topics
# (revenue, vendor costs, gross margin). The embedding model sees them
# as nearly identical vectors — so it mixes up Q1 and Q4 answers.
#
# Solution: Add "[Document: file.md] [Period: Q1 2024]" to the START of
# every chunk. Now the embedding encodes WHICH quarter it belongs to,
# making Q1 and Q4 vectors genuinely different.

def load_documents(folders: list, base_path: str) -> list:
    documents = []

    for folder in folders:
        folder_path = os.path.join(base_path, folder)

        # Skip if folder doesn't exist
        if not os.path.exists(folder_path):
            print(f"[WARNING] Folder not found: {folder_path}")
            continue

        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)

            try:
                # Pick the right loader based on file extension
                if fname.endswith(".md") or fname.endswith(".txt"):
                    loader = TextLoader(fpath, encoding="utf-8")
                elif fname.endswith(".csv"):
                    loader = CSVLoader(fpath, encoding="utf-8")
                elif fname.endswith(".pdf"):
                    loader = PyPDFLoader(fpath)
                elif fname.endswith(".docx"):
                    loader = Docx2txtLoader(fpath)
                else:
                    continue  # skip unsupported files silently

                # Load the file — returns a list of Document objects
                docs = loader.load()

                # Get quarter/year/doc_type from the filename
                tags = extract_file_tags(fname)
                period = f"{tags['quarter']} {tags['year']}".strip()

                # Build the metadata header that goes at the top of every chunk
                header = f"[Document: {fname}] [Type: {tags['doc_type']}] [Period: {period}]\n\n"

                for doc in docs:
                    # Save metadata so we can show sources in the answer
                    doc.metadata["source"]   = fname
                    doc.metadata["folder"]   = folder
                    doc.metadata["quarter"]  = tags["quarter"]
                    doc.metadata["year"]     = tags["year"]
                    doc.metadata["doc_type"] = tags["doc_type"]

                    # Prepend header to chunk text — this is the collision fix
                    doc.page_content = header + doc.page_content

                documents.extend(docs)

            except Exception as e:
                # If one file fails, skip it and continue with the rest
                print(f"[ERROR] Could not load {fpath}: {e}")

    return documents
