"""
Answer Generator Module for RAG System.
Handles response generation using retrieved context.
"""

from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

class AnswerGenerator:
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the Answer Generator.
        
        Args:
            model_name: Name of the model to use for answer generation
            reranker_model: Name of the model to use for reranking
            device: Device to run the models on ('cuda', 'cpu', etc.)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize reranker
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model)
        self.reranker_model.to(self.device)
        
        # Initialize generator
        self.generator = pipeline(
            'text2text-generation',
            model=model_name,
            device=self.device if self.device != 'cpu' else -1
        )
    
    def rerank_passages(
        self,
        query: str,
        passages: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank passages based on relevance to query.
        
        Args:
            query: Original query string
            passages: List of passages with their metadata
            top_k: Number of top passages to return
            
        Returns:
            List of reranked passages
        """
        scores = []
        
        for passage in passages:
            # Prepare input
            inputs = self.reranker_tokenizer(
                query,
                passage['content'],
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get relevance score
            with torch.no_grad():
                output = self.reranker_model(**inputs)
                score = output.logits[0].item()
            
            # Add score to passage metadata
            passage_with_score = passage.copy()
            passage_with_score['relevance_score'] = score
            scores.append(passage_with_score)
        
        # Sort by score and return top k
        reranked = sorted(scores, key=lambda x: x['relevance_score'], reverse=True)
        return reranked[:top_k]
    
    def generate_answer(
        self,
        query: str,
        contexts: List[Dict[str, Any]],
        max_length: int = 512
    ) -> Dict[str, Any]:
        """
        Generate an answer using the query and retrieved contexts.
        
        Args:
            query: Original query string
            contexts: List of relevant context passages
            max_length: Maximum length of generated answer
            
        Returns:
            Dictionary containing the answer and metadata
        """
        # Combine contexts into a single string
        context_text = " ".join([c['content'] for c in contexts])
        
        # Prepare prompt
        prompt = f"""Answer the following question using the provided context. If the context doesn't contain enough information, say "I don't have enough information to answer this question."

Context: {context_text}

Question: {query}

Answer:"""
        
        # Generate answer
        response = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )[0]['generated_text']
        
        # Prepare response with metadata
        result = {
            'answer': response,
            'sources': [
                {
                    'content': ctx['content'],
                    'relevance_score': ctx.get('relevance_score', 0),
                    'metadata': {k: v for k, v in ctx.items() if k not in ['content', 'relevance_score']}
                }
                for ctx in contexts
            ],
            'query': query
        }
        
        return result
    
    def process_query(
        self,
        query: str,
        retrieved_passages: List[Dict[str, Any]],
        top_k_rerank: int = 3
    ) -> Dict[str, Any]:
        """
        Process a query end-to-end: rerank passages and generate answer.
        
        Args:
            query: Original query string
            retrieved_passages: List of retrieved passages
            top_k_rerank: Number of passages to keep after reranking
            
        Returns:
            Dictionary containing answer and supporting information
        """
        # Rerank passages
        reranked_passages = self.rerank_passages(query, retrieved_passages, top_k=top_k_rerank)
        
        # Generate answer
        result = self.generate_answer(query, reranked_passages)
        
        return result