"""
Configuration settings for SPRNS system.
"""

import os
from typing import Dict, Any

class Config:
    """Configuration class for SPRNS system."""
    
    # Neo4j Configuration
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
    NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
    
    # Model Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    GENERATION_MODEL = os.getenv("GENERATION_MODEL", "google/flan-t5-base")
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    # Retrieval Configuration
    MAX_RETRIEVED_DOCS = int(os.getenv("MAX_RETRIEVED_DOCS", "10"))
    TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))
    MAX_ANSWER_LENGTH = int(os.getenv("MAX_ANSWER_LENGTH", "512"))
    
    # Dashboard Configuration
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8501"))
    DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "localhost")
    
    @classmethod
    def get_neo4j_config(cls) -> Dict[str, Any]:
        """Get Neo4j configuration as dictionary."""
        return {
            "uri": cls.NEO4J_URI,
            "username": cls.NEO4J_USERNAME,
            "password": cls.NEO4J_PASSWORD,
            "database": cls.NEO4J_DATABASE
        }
    
    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration as dictionary."""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "generation_model": cls.GENERATION_MODEL,
            "reranker_model": cls.RERANKER_MODEL
        }
    
    @classmethod
    def get_retrieval_config(cls) -> Dict[str, Any]:
        """Get retrieval configuration as dictionary."""
        return {
            "max_retrieved_docs": cls.MAX_RETRIEVED_DOCS,
            "top_k_rerank": cls.TOP_K_RERANK,
            "max_answer_length": cls.MAX_ANSWER_LENGTH
        }