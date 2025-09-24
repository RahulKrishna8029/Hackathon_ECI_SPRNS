"""
Tests for the Query Processing Module.
"""

import pytest
from unittest.mock import MagicMock, patch
import json
from retrieval.core.query_processor import QueryProcessor

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        client = MagicMock()
        mock.return_value = client
        yield client

@pytest.fixture
def query_processor(mock_openai):
    return QueryProcessor(api_key="test-key")

def test_query_processor_initialization():
    with pytest.raises(ValueError):
        # Should raise error if no API key provided
        QueryProcessor()

    # Should initialize with provided API key
    processor = QueryProcessor(api_key="test-key")
    assert processor.model == "gpt-4-turbo-preview"
    assert processor.embedding_model == "text-embedding-3-large"

def test_generate_embeddings(query_processor, mock_openai):
    # Mock embedding response
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.embeddings.create.return_value = mock_response

    embeddings = query_processor.generate_embeddings("test query")
    
    assert isinstance(embeddings, list)
    assert len(embeddings) == 3
    assert embeddings == [0.1, 0.2, 0.3]

def test_classify_intent(query_processor, mock_openai):
    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="search"))
    ]
    mock_openai.chat.completions.create.return_value = mock_response

    intent = query_processor.classify_intent("what is a knowledge graph?")
    
    assert isinstance(intent, str)
    assert intent == "search"

def test_extract_parameters(query_processor, mock_openai):
    # Mock completion response
    mock_parameters = {
        "filters": {"type": "academic"},
        "sort": "relevance",
        "limit": 5,
        "time_range": "last_year",
        "categories": ["computer_science"],
        "metadata": {}
    }
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(mock_parameters)))
    ]
    mock_openai.chat.completions.create.return_value = mock_response

    params = query_processor.extract_parameters(
        "Show me 5 academic papers about knowledge graphs from last year"
    )
    
    assert isinstance(params, dict)
    assert "filters" in params
    assert "sort" in params
    assert "limit" in params
    assert params["limit"] == 5
    assert params["time_range"] == "last_year"

def test_process_query(query_processor, mock_openai):
    # Mock embedding response
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
    mock_openai.embeddings.create.return_value = mock_embedding_response

    # Mock intent response
    mock_intent_response = MagicMock()
    mock_intent_response.choices = [MagicMock(message=MagicMock(content="search"))]
    
    # Mock parameters response
    mock_params = {
        "filters": {},
        "sort": None,
        "limit": 10,
        "time_range": None,
        "categories": [],
        "metadata": {}
    }
    mock_params_response = MagicMock()
    mock_params_response.choices = [
        MagicMock(message=MagicMock(content=json.dumps(mock_params)))
    ]

    # Set up sequential responses
    mock_openai.chat.completions.create.side_effect = [
        mock_intent_response,
        mock_params_response
    ]

    result = query_processor.process_query("test query")
    
    assert isinstance(result, dict)
    assert "raw_query" in result
    assert "embeddings" in result
    assert "intent" in result
    assert "parameters" in result
    assert result["raw_query"] == "test query"
    assert result["intent"] == "search"