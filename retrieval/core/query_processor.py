"""
Query Processing Module for RAG System.
Handles query processing using GPT-4.5 Turbo and OpenAI embeddings.
"""

from typing import Dict, Any, Optional, List
import os
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class QueryProcessor:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo-preview",
        embedding_model: str = "text-embedding-3-large",
        max_tokens: int = 4096
    ):
        """
        Initialize the Query Processor with GPT-4.5 Turbo.
        
        Args:
            api_key: OpenAI API key (optional if set in environment)
            model: GPT model to use
            embedding_model: OpenAI embedding model to use
            max_tokens: Maximum tokens for response
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        if not self.client.api_key:
            raise ValueError("OpenAI API key must be provided either directly or via OPENAI_API_KEY environment variable")
        
        self.model = model
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a raw query using GPT-4.5 Turbo.
        
        Args:
            query: Raw query string
            
        Returns:
            Dict containing processed query information
        """
        # Generate embeddings using OpenAI
        embeddings = self.generate_embeddings(query)
        
        # Use GPT-4.5 Turbo for intent classification
        intent = self.classify_intent(query)
        
        # Extract parameters using GPT-4.5 Turbo
        parameters = self.extract_parameters(query)
        
        return {
            "raw_query": query,
            "embeddings": embeddings,
            "intent": intent,
            "parameters": parameters
        }
    
    def generate_embeddings(self, text: str) -> List[float]:
        """
        Generate embeddings using OpenAI's embedding model.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding
    
    def classify_intent(self, query: str) -> str:
        """
        Classify query intent using GPT-4.5 Turbo.
        
        Args:
            query: Raw query string
            
        Returns:
            Classified intent string
        """
        system_prompt = """
        Analyze the query and classify its intent into one of these categories:
        - search: Looking for information
        - compare: Comparing multiple items
        - analyze: Deep analysis request
        - summarize: Summary request
        Return only the category name.
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=20,
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    
    def extract_parameters(self, query: str) -> Dict[str, Any]:
        """
        Extract parameters using GPT-4.5 Turbo.
        
        Args:
            query: Raw query string
            
        Returns:
            Dict of extracted parameters
        """
        system_prompt = """
        Extract structured parameters from the query. Return a JSON object with:
        - filters: Any filtering criteria
        - sort: Sorting preferences
        - limit: Number of results (default 10)
        - time_range: Any time constraints
        - categories: Any mentioned categories or topics
        - metadata: Any additional query metadata
        """
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            max_tokens=200,
            temperature=0,
            response_format={ "type": "json_object" }
        )
        
        # Parse the response as parameters
        try:
            import json
            parameters = json.loads(response.choices[0].message.content)
        except:
            parameters = {
                "filters": {},
                "sort": None,
                "limit": 10,
                "time_range": None,
                "categories": [],
                "metadata": {}
            }
            
        return parameters