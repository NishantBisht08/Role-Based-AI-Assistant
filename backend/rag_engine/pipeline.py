# ── RAG Pipeline ──────────────────────────────────────────────────────────────
# This is the main orchestrator that ties everything together.
# It runs the full RAG pipeline:
#   Check role → Get DB → Search docs → Build prompt → Get answer
# This function is called by FastAPI when a user asks a question.
# ──────────────────────────────────────────────────────────────────────────────

import os

# Loads environment variables from .env file
from dotenv import load_dotenv

# Groq client — used to call the LLaMA model for generating answers
from groq import Groq

from .rbac import ROLE_FOLDERS, enforce_rbac
from .vectorstore import get_or_build_vectorstore

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, "..", "..", ".env"))


# ── Step 8: Main function — tie everything together ───────────────────────────
# This is the function called by FastAPI when a user asks a question.
# It runs the full RAG pipeline:
#   Check role → Get DB → Search docs → Build prompt → Get answer

def ask_question(role: str, query: str, debug: bool = False) -> dict:

    BASE_PATH = os.path.join(BASE_DIR, "..", "..", "data")  # where all the data folders live

    # --- 1. Check if the role is valid ---
    is_valid, error_msg = enforce_rbac(role)
    if not is_valid:
        # Return error immediately — no DB or LLM work wasted
        return {"answer": error_msg, "sources": []}

    # Normalize role — "Finance " becomes "finance"
    role = role.strip().lower()

    # Get the list of folders this role is allowed to read
    allowed_folders = ROLE_FOLDERS[role]

    # --- 2. Get the ChromaDB for this role ---
    try:
        db = get_or_build_vectorstore(role, allowed_folders, BASE_PATH)
    except ValueError as e:
        return {"answer": str(e), "sources": []}

    # --- 3. Search for relevant chunks ---
    # MMR = Maximal Marginal Relevance
    # fetch_k=30: first fetch 30 similar chunks
    # k=10: then pick the 10 most DIVERSE ones from those 30
    # This prevents getting 10 chunks all from the same document
    retrieved_docs = db.max_marginal_relevance_search(query, k=10, fetch_k=30)

    if not retrieved_docs:
        return {"answer": "I do not have access to that information.", "sources": []}

    # --- 4. Build context from retrieved chunks ---
    # Join all chunks into one big string with source labels
    # Example: "[Source: quarterly_financial_report.md]\n...content..."
        # 1. Create an empty list to hold our formatted chunks
    formatted_chunks = []

    # 2. Loop through the documents normally
    for doc in retrieved_docs:
        
        # 3. Get the filename safely
        filename = doc.metadata.get('source', 'unknown')
        
        # 4. Format the text block
        text_block = f"[Source: {filename}]\n{doc.page_content}"
        
        # 5. Add it to our list
        formatted_chunks.append(text_block)

    # 6. Glue the whole list together using our visual divider
    context = "\n\n---\n\n".join(formatted_chunks)

    # Get unique source filenames for the response
    # 1. Create an empty list to hold our source filenames
    sources = []

    # 2. Loop through our 10 relevant document chunks
    for doc in retrieved_docs:
        
        # 3. Get the filename safely
        filename = doc.metadata.get("source", "unknown")
        
        # 4. Check if this filename is ALREADY in our list
        if filename not in sources:
            # 5. If it is brand new, add it to the list
            sources.append(filename)

    # 6. Alphabetize the final list so it looks nice for the user
    sources.sort()

    # --- 5. Build the prompt and call LLaMA ---
    # The prompt tells LLaMA exactly what to do and what NOT to do
    prompt = f"""You are a secure enterprise AI assistant for FinSolve Technologies.

RULES:
- Answer ONLY using the context provided below.
- If the answer is not in the context, say: "I do not have access to that information."
- Never guess or make up data.
- Be concise and professional.
- Always mention the source document name when referencing data.

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    try:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # LLaMA 70B model on Groq
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,    # low = more factual, less creative
            max_tokens=1024,    # max length of the answer (~750 words)
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"LLM call failed: {e}"

    # --- 6. Return answer + sources ---
    return {"answer": answer, "sources": sources}
