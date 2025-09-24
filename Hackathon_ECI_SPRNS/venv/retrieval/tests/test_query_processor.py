"""
Tests for the Query Processing Module.
"""

import pytest
import torch
from retrieval.core.query_processor import QueryProcessor

@pytest.fixture
def query_processor():
    return QueryProcessor()

def test_query_processor_initialization(query_processor):
    assert query_processor is not None
    assert isinstance(query_processor, QueryProcessor)

def test_generate_embeddings(query_processor):
    query = "Test query for embedding generation"
    embeddings = query_processor.generate_embeddings(query)
    
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.dim() == 1  # Should be a 1D tensor
    assert embeddings.shape[0] == 384  # Default embedding dimension for MiniLM

def test_process_query(query_processor):
    query = "What is the capital of France?"
    result = query_processor.process_query(query)
    
    assert isinstance(result, dict)
    assert "raw_query" in result
    assert "embeddings" in result
    assert "intent" in result
    assert "parameters" in result
    
    assert result["raw_query"] == query
    assert isinstance(result["embeddings"], torch.Tensor)
    assert isinstance(result["intent"], str)
    assert isinstance(result["parameters"], dict)

def test_extract_parameters(query_processor):
    query = "Show me the top 10 documents about machine learning"
    params = query_processor.extract_parameters(query)
    
    assert isinstance(params, dict)
    assert "filters" in params
    assert "sort" in params
    assert "limit" in params