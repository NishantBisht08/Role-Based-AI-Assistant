import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader

from langchain_community.vectorstores import Chroma


from langchain_community.embeddings import HuggingFaceEmbeddings

from groq import Groq

load_dotenv()

# -----------------------------
# 1. Load document
# -----------------------------
loader = TextLoader("../data/general/employee_handbook.md", encoding="utf-8")
documents = loader.load()

# -----------------------------
# 2. Create embeddings & vector DB
# -----------------------------
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embedding)

# -----------------------------
# 3. Ask question
# -----------------------------
query = "How many sick leaves are allowed?"

retrieved_docs = db.similarity_search(query, k=3)

context = "\n\n".join([doc.page_content for doc in retrieved_docs])

# -----------------------------
# 4. Call Llama 3 via Groq
# -----------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

prompt = f"""
You are an internal company assistant.
Answer the question ONLY using the context below.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0.2
)

print("\n Answer:\n")
print(response.choices[0].message.content)
print("\n Sources:\n")
for doc in retrieved_docs:
    print(doc.metadata.get("source"))