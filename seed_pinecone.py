import os
import sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# We need the PINECONE_API_KEY
if not os.environ.get("PINECONE_API_KEY"):
    print("ERROR: Please set PINECONE_API_KEY in your .env file!")
    sys.exit(1)

from backend.rag_engine.rbac import ROLE_FOLDERS
from backend.rag_engine.document_loader import load_documents
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
index_name = os.environ.get("PINECONE_INDEX_NAME", "company-docs")

print(f"Initializing Pinecone and FastEmbed...")
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Ensure index exists
if index_name not in pc.list_indexes().names():
    print(f"ERROR: Pinecone index '{index_name}' does not exist.")
    print(f"Please create a Serverless Index named '{index_name}' with Dimensions: 384, Metric: Cosine on the Pinecone dashboard.")
    sys.exit(1)

embedding_model = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    threads=1
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

for role, folders in ROLE_FOLDERS.items():
    print(f"\n--- Seeding for role: {role} ---")
    documents = load_documents(folders, BASE_PATH)
    
    if not documents:
        print(f"No documents found for role '{role}'. Skipping.")
        continue
        
    print(f"Loaded {len(documents)} document pages. Chunking...")
    chunks = text_splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks. Uploading to Pinecone (namespace: {role})...")
    
    PineconeVectorStore.from_documents(
        chunks, 
        embedding_model, 
        index_name=index_name, 
        namespace=role
    )
    
    print(f"Successfully seeded {role}!")

print("\nAll roles seeded successfully!")
