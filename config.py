"""
Central configuration for the RAG Q&A Chatbot.
Keeping all tunable parameters in one place makes it easy to run
chunking/model experiments (per the "AI Tool Evaluation" workflow).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
DATA_DIR = "data"                     # raw PDFs go here
VECTORSTORE_DIR = "vectorstore"       # FAISS index persisted here
LOG_DB_PATH = "logs/query_logs.db"    # SQLite database for monitoring

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Chunking (the parameters you sweep during evaluation) ---
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

# --- Retrieval ---
RETRIEVER_TOP_K = 4

# --- LLM (Groq-hosted models) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL_NAME = "openai/gpt-oss-20b"  # use a model ID currently available on this Groq account
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 1024

# --- Prompt ---
SYSTEM_PROMPT = """You are a helpful assistant answering questions strictly using the
provided context retrieved from the user's documents. If the answer is not
contained in the context, say you don't know rather than guessing.
Always be concise and cite which part of the context you used when relevant.

Context:
{context}
"""
