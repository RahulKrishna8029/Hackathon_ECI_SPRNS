"""
Tests for the Answer Generator Module.
"""

import pytest
from retrieval.core.answer_generator import AnswerGenerator

@pytest.fixture
def answer_generator():
    return AnswerGenerator()

@pytest.fixture
def sample_passages():
    return [
        {
            'content': 'Paris is the capital of France.',
            'id': 1,
            'source': 'wiki'
        },
        {
            'content': 'France is a country in Western Europe.',
            'id': 2,
            'source': 'wiki'
        },
        {
            'content': 'The Eiffel Tower is located in Paris.',
            'id': 3,
            'source': 'wiki'
        }
    ]

def test_answer_generator_initialization(answer_generator):
    assert answer_generator is not None
    assert answer_generator.reranker_model is not None
    assert answer_generator.generator is not None

def test_rerank_passages(answer_generator, sample_passages):
    query = "What is the capital of France?"
    reranked = answer_generator.rerank_passages(query, sample_passages, top_k=2)
    
    assert len(reranked) == 2
    assert all('relevance_score' in p for p in reranked)
    assert reranked[0]['relevance_score'] >= reranked[1]['relevance_score']

def test_generate_answer(answer_generator, sample_passages):
    query = "What is the capital of France?"
    result = answer_generator.generate_answer(query, sample_passages)
    
    assert isinstance(result, dict)
    assert 'answer' in result
    assert 'sources' in result
    assert 'query' in result
    assert len(result['sources']) == len(sample_passages)

def test_process_query(answer_generator, sample_passages):
    query = "What is the capital of France?"
    result = answer_generator.process_query(query, sample_passages)
    
    assert isinstance(result, dict)
    assert 'answer' in result
    assert 'sources' in result
    assert 'query' in result
    assert len(result['sources']) == 3  # top_k_rerank default is 3