"""
Tests for the Neo4j Connector Module.
"""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.utils.neo4j_connector import Neo4jConnector

@pytest.fixture
def mock_neo4j():
    with patch('retrieval.utils.neo4j_connector.GraphDatabase') as mock:
        # Mock the driver
        mock_driver = MagicMock()
        mock_driver.verify_connectivity = MagicMock()
        mock.driver.return_value = mock_driver
        
        # Mock session and transaction
        mock_session = MagicMock()
        mock_driver.session.return_value = mock_session
        
        yield mock

@pytest.fixture
def neo4j_connector(mock_neo4j):
    return Neo4jConnector(
        uri="bolt://localhost:7687",
        username="neo4j",
        password="password"
    )

def test_neo4j_connector_initialization(mock_neo4j, neo4j_connector):
    assert neo4j_connector is not None
    mock_neo4j.driver.assert_called_once_with(
        "bolt://localhost:7687",
        auth=("neo4j", "password")
    )

def test_add_document(neo4j_connector):
    document = {
        'id': '123',
        'title': 'Test Document',
        'content': 'Test content'
    }
    
    neo4j_connector.add_document(document)
    
    # Verify query execution
    session = neo4j_connector.driver.session.return_value
    session.run.assert_called_once()
    call_args = session.run.call_args[0]
    
    # Verify query contains MERGE statement
    assert 'MERGE' in call_args[0]
    assert 'Document' in call_args[0]

def test_add_relationship(neo4j_connector):
    neo4j_connector.add_relationship(
        from_id='123',
        to_id='456',
        relationship_type='RELATES_TO',
        properties={'weight': 0.8}
    )
    
    # Verify query execution
    session = neo4j_connector.driver.session.return_value
    session.run.assert_called_once()
    call_args = session.run.call_args[0]
    
    # Verify query contains MERGE relationship
    assert 'MERGE' in call_args[0]
    assert 'RELATES_TO' in call_args[0]

def test_get_document_by_id(neo4j_connector):
    # Mock return value
    mock_result = [{'d': {'id': '123', 'title': 'Test'}}]
    session = neo4j_connector.driver.session.return_value
    session.run.return_value = [MagicMock(dict=lambda: record) for record in mock_result]
    
    result = neo4j_connector.get_document_by_id('123')
    
    assert result == {'id': '123', 'title': 'Test'}
    session.run.assert_called_once()

def test_get_related_documents(neo4j_connector):
    # Mock return value
    mock_result = [
        {'related': {'id': '456', 'title': 'Related Doc 1'}},
        {'related': {'id': '789', 'title': 'Related Doc 2'}}
    ]
    session = neo4j_connector.driver.session.return_value
    session.run.return_value = [MagicMock(dict=lambda: record) for record in mock_result]
    
    results = neo4j_connector.get_related_documents('123', relationship_type='RELATES_TO')
    
    assert len(results) == 2
    session.run.assert_called_once()

def test_search_documents(neo4j_connector):
    # Mock return value
    mock_result = [
        {'d': {'id': '123', 'title': 'Test 1'}},
        {'d': {'id': '456', 'title': 'Test 2'}}
    ]
    session = neo4j_connector.driver.session.return_value
    session.run.return_value = [MagicMock(dict=lambda: record) for record in mock_result]
    
    results = neo4j_connector.search_documents({'title': 'Test'})
    
    assert len(results) == 2
    session.run.assert_called_once()