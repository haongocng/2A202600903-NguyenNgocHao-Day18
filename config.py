"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
# Dynamically route OpenAI client to TokenRouter (Minimax) if present, otherwise direct OpenAI
if os.getenv("MINIMAX_API_KEY") and os.getenv("MINIMAX_API_KEY") != "sk-":
    OPENAI_API_KEY = os.getenv("MINIMAX_API_KEY", "")
    OPENAI_BASE_URL = os.getenv("MINIMAX_BASE_URL", "https://api.tokenrouter.com/v1")
    OPENAI_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M3")
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL = ""
    OPENAI_MODEL = "gpt-4o-mini"


COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

# --- Weaviate / Qdrant ---
# We keep these for interface compatibility, but we will route index/search to Weaviate Cloud
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "embed-multilingual-v3.0"
EMBEDDING_DIM = 1024


# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
