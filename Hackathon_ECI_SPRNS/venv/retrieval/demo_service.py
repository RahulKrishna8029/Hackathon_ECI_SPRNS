"""
Demo Retrieval Service for SPRNS System.
Lightweight version that works without heavy ML dependencies.
"""

import time
import random
from typing import List, Dict, Any

class DemoRetrievalService:
    """
    Demo version of the retrieval service that simulates RAG functionality
    without requiring heavy ML dependencies.
    """
    
    def __init__(self):
        """Initialize the demo service."""
        self.knowledge_base = self._create_demo_knowledge_base()
        print("Demo Retrieval Service initialized successfully!")
    
    def _create_demo_knowledge_base(self) -> List[Dict[str, Any]]:
        """Create a demo knowledge base with sample documents."""
        return [
            {
                'id': 'doc_1',
                'title': 'Introduction to Machine Learning',
                'content': 'Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications.',
                'category': 'AI/ML',
                'keywords': ['machine learning', 'AI', 'algorithms', 'data', 'patterns']
            },
            {
                'id': 'doc_2',
                'title': 'Deep Learning Fundamentals',
                'content': 'Deep learning is a specialized branch of machine learning that uses neural networks with multiple layers to model and understand complex patterns in data. It has revolutionized fields like computer vision, natural language processing, and speech recognition.',
                'category': 'AI/ML',
                'keywords': ['deep learning', 'neural networks', 'computer vision', 'NLP']
            },
            {
                'id': 'doc_3',
                'title': 'Natural Language Processing',
                'content': 'Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers and human language. It involves techniques for analyzing, understanding, and generating human language in a valuable way.',
                'category': 'AI/ML',
                'keywords': ['NLP', 'language', 'text processing', 'linguistics']
            },
            {
                'id': 'doc_4',
                'title': 'Data Science Overview',
                'content': 'Data science is an interdisciplinary field that combines statistics, mathematics, programming, and domain expertise to extract insights from data. It involves data collection, cleaning, analysis, and visualization to support decision-making.',
                'category': 'Data Science',
                'keywords': ['data science', 'statistics', 'analysis', 'visualization']
            },
            {
                'id': 'doc_5',
                'title': 'Python Programming',
                'content': 'Python is a high-level, interpreted programming language known for its simplicity and readability. It is widely used in web development, data science, artificial intelligence, and automation due to its extensive library ecosystem.',
                'category': 'Programming',
                'keywords': ['python', 'programming', 'web development', 'automation']
            },
            {
                'id': 'doc_6',
                'title': 'Database Systems',
                'content': 'Database systems are organized collections of data that are stored and accessed electronically. They provide efficient ways to store, retrieve, and manage large amounts of information using structured query languages like SQL.',
                'category': 'Database',
                'keywords': ['database', 'SQL', 'data storage', 'queries']
            }
        ]
    
    def _calculate_relevance(self, query: str, document: Dict[str, Any]) -> float:
        """Calculate a simple relevance score between query and document."""
        query_words = set(query.lower().split())
        
        # Check title relevance
        title_words = set(document['title'].lower().split())
        title_overlap = len(query_words.intersection(title_words))
        
        # Check content relevance
        content_words = set(document['content'].lower().split())
        content_overlap = len(query_words.intersection(content_words))
        
        # Check keyword relevance
        keyword_overlap = sum(1 for keyword in document['keywords'] 
                            if any(word in keyword.lower() for word in query_words))
        
        # Calculate weighted score
        score = (title_overlap * 0.4 + content_overlap * 0.4 + keyword_overlap * 0.2)
        
        # Add some randomness to simulate embedding similarity
        score += random.uniform(0, 0.1)
        
        return min(score, 1.0)
    
    def retrieve_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query."""
        # Calculate relevance scores
        scored_docs = []
        for doc in self.knowledge_base:
            relevance = self._calculate_relevance(query, doc)
            if relevance > 0.1:  # Only include somewhat relevant documents
                doc_with_score = doc.copy()
                doc_with_score['relevance_score'] = relevance
                scored_docs.append(doc_with_score)
        
        # Sort by relevance and return top results
        scored_docs.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_docs[:limit]
    
    def generate_answer(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """Generate a simple answer based on retrieved documents."""
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Use the most relevant document as the primary source
        primary_doc = documents[0]
        
        # Create a simple answer template
        answer_templates = [
            f"Based on the available information, {primary_doc['content'][:200]}...",
            f"According to the documentation, {primary_doc['content'][:200]}...",
            f"From what I found, {primary_doc['content'][:200]}...",
        ]
        
        answer = random.choice(answer_templates)
        
        # Add information about additional sources if available
        if len(documents) > 1:
            answer += f"\n\nI found {len(documents)} relevant sources that discuss this topic."
        
        return answer
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process a user query end-to-end."""
        try:
            # Simulate processing time
            time.sleep(0.5)
            
            # Retrieve relevant documents
            documents = self.retrieve_documents(query)
            
            if not documents:
                return {
                    'answer': "I couldn't find any relevant documents to answer your question. Try rephrasing your query or asking about topics like machine learning, data science, or programming.",
                    'sources': [],
                    'query': query,
                    'status': 'no_results'
                }
            
            # Generate answer
            answer = self.generate_answer(query, documents)
            
            # Format sources for display
            sources = []
            for doc in documents:
                sources.append({
                    'content': doc['content'],
                    'relevance_score': doc['relevance_score'],
                    'metadata': {
                        'title': doc['title'],
                        'category': doc['category'],
                        'id': doc['id']
                    }
                })
            
            return {
                'answer': answer,
                'sources': sources,
                'query': query,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'sources': [],
                'query': query,
                'status': 'error'
            }
    
    def close(self):
        """Close the service (no-op for demo version)."""
        pass