"""
Configuration settings for the retrieval system.
"""
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384
TOP_K_RESULTS = 5
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Neo4j settings
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Vector store settings
FAISS_INDEX_PATH = BASE_DIR / "data" / "faiss_index"

# Logging settings
LOG_LEVEL = "INFO"