"""
Main Retrieval Service for RAG System.
Integrates query processing, document retrieval, and answer generation.
"""

import os
import sys
from typing import List, Dict, Any, Optional
from retrieval.core.query_processor import QueryProcessor
from retrieval.core.answer_generator import AnswerGenerator
from retrieval.utils.neo4j_connector import Neo4jConnector

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import Config
except ImportError:
    # Fallback if config is not available
    class Config:
        NEO4J_URI = "bolt://localhost:7687"
        NEO4J_USERNAME = "neo4j"
        NEO4J_PASSWORD = "password"
        EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
        GENERATION_MODEL = "google/flan-t5-base"

class RetrievalService:
    def __init__(
        self,
        neo4j_uri: str = None,
        neo4j_username: str = None,
        neo4j_password: str = None,
        embedding_model: str = None,
        generation_model: str = None
    ):
        """
        Initialize the Retrieval Service.
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_username: Neo4j username
            neo4j_password: Neo4j password
            embedding_model: Model for query embeddings
            generation_model: Model for answer generation
        """
        # Use config defaults if not provided
        neo4j_uri = neo4j_uri or Config.NEO4J_URI
        neo4j_username = neo4j_username or Config.NEO4J_USERNAME
        neo4j_password = neo4j_password or Config.NEO4J_PASSWORD
        embedding_model = embedding_model or Config.EMBEDDING_MODEL
        generation_model = generation_model or Config.GENERATION_MODEL
        
        # Initialize components
        self.query_processor = QueryProcessor(embedding_model=embedding_model)
        self.answer_generator = AnswerGenerator(model_name=generation_model)
        
        # Initialize Neo4j connector (with error handling and mock fallback)
        try:
            self.neo4j_connector = Neo4jConnector(
                uri=neo4j_uri,
                username=neo4j_username,
                password=neo4j_password
            )
            print("Connected to Neo4j database")
        except Exception as e:
            print(f"Warning: Could not connect to Neo4j: {e}")
            print("Falling back to mock Neo4j connector with sample customer data")
            try:
                # Import and use mock connector
                sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from mock_graph.mock_neo4j import MockNeo4jConnector
                self.neo4j_connector = MockNeo4jConnector(
                    uri=neo4j_uri,
                    username=neo4j_username,
                    password=neo4j_password
                )
            except Exception as mock_error:
                print(f"Error initializing mock connector: {mock_error}")
                self.neo4j_connector = None
    
    def retrieve_documents(
        self,
        query: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents based on the query.
        
        Args:
            query: User query string
            limit: Maximum number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        if not self.neo4j_connector:
            # Return mock documents if Neo4j is not available
            return self._get_mock_documents(query, limit)
        
        try:
            # For now, do a simple text search in Neo4j
            # In a real implementation, you'd use vector similarity search
            results = self.neo4j_connector.execute_query(
                """
                MATCH (d:Document)
                WHERE d.content CONTAINS $query_text
                RETURN d
                LIMIT $limit
                """,
                {'query_text': query, 'limit': limit}
            )
            
            documents = []
            for record in results:
                doc = record['d']
                documents.append({
                    'content': doc.get('content', ''),
                    'title': doc.get('title', 'Untitled'),
                    'id': doc.get('id', ''),
                    'metadata': {k: v for k, v in doc.items() if k not in ['content', 'title', 'id']}
                })
            
            return documents
            
        except Exception as e:
            print(f"Error retrieving from Neo4j: {e}")
            return self._get_mock_documents(query, limit)
    
    def _get_mock_documents(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """
        Return mock documents for testing when Neo4j is not available.
        This method is now deprecated as we use MockNeo4jConnector instead.
        
        Args:
            query: User query string
            limit: Maximum number of documents to return
            
        Returns:
            List of mock documents
        """
        # This is kept for backward compatibility but should not be used
        # when MockNeo4jConnector is available
        mock_docs = [
            {
                'content': f'This is a fallback document about {query}. The mock Neo4j connector should be used instead.',
                'title': f'Fallback Document: {query}',
                'id': 'fallback_doc_1',
                'metadata': {'source': 'fallback', 'confidence': 0.5}
            }
        ]
        
        return mock_docs[:limit]
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query end-to-end.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary containing the answer and supporting information
        """
        try:
            # Process the query
            processed_query = self.query_processor.process_query(query)
            
            # Retrieve relevant documents
            documents = self.retrieve_documents(query)
            
            if not documents:
                return {
                    'answer': "I couldn't find any relevant documents to answer your question.",
                    'sources': [],
                    'query': query,
                    'status': 'no_results'
                }
            
            # Generate answer using retrieved documents
            result = self.answer_generator.process_query(query, documents)
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'sources': [],
                'query': query,
                'status': 'error'
            }
    
    def close(self):
        """Close database connections."""
        if self.neo4j_connector:
            self.neo4j_connector.close()