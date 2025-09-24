"""
Neo4j Connector Module for RAG System.
Handles graph database operations and queries.
"""

from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

class Neo4jConnector:
    def __init__(
        self,
        uri: str,
        username: str,
        password: str,
        database: str = "neo4j"
    ):
        """
        Initialize the Neo4j connector.
        
        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Name of the database to use
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        
        # Verify connection
        try:
            self.driver.verify_connectivity()
        except Neo4jError as e:
            raise ConnectionError(f"Failed to connect to Neo4j: {e}")
    
    def close(self):
        """Close the database connection."""
        self.driver.close()
    
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.
        
        Args:
            query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of query results
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [dict(record) for record in result]
    
    def add_document(self, document: Dict[str, Any]) -> None:
        """
        Add a document to the graph database.
        
        Args:
            document: Document dictionary containing metadata and content
        """
        query = """
        MERGE (d:Document {id: $id})
        SET d += $properties
        """
        
        # Separate id from other properties
        doc_id = document.get('id')
        if not doc_id:
            raise ValueError("Document must have an 'id' field")
        
        properties = {k: v for k, v in document.items() if k != 'id'}
        
        self.execute_query(query, {
            'id': doc_id,
            'properties': properties
        })
    
    def add_relationship(
        self,
        from_id: str,
        to_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a relationship between two documents.
        
        Args:
            from_id: ID of the source document
            to_id: ID of the target document
            relationship_type: Type of relationship
            properties: Optional relationship properties
        """
        query = f"""
        MATCH (a:Document {{id: $from_id}})
        MATCH (b:Document {{id: $to_id}})
        MERGE (a)-[r:{relationship_type}]->(b)
        SET r += $properties
        """
        
        self.execute_query(query, {
            'from_id': from_id,
            'to_id': to_id,
            'properties': properties or {}
        })
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document dictionary if found, None otherwise
        """
        query = """
        MATCH (d:Document {id: $id})
        RETURN d
        """
        
        results = self.execute_query(query, {'id': doc_id})
        return results[0]['d'] if results else None
    
    def get_related_documents(
        self,
        doc_id: str,
        relationship_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get documents related to the given document.
        
        Args:
            doc_id: Document ID
            relationship_type: Optional specific relationship type to look for
            limit: Maximum number of related documents to return
            
        Returns:
            List of related documents
        """
        if relationship_type:
            query = f"""
            MATCH (d:Document {{id: $id}})-[r:{relationship_type}]->(related:Document)
            RETURN related
            LIMIT $limit
            """
        else:
            query = """
            MATCH (d:Document {id: $id})-[r]->(related:Document)
            RETURN related
            LIMIT $limit
            """
        
        results = self.execute_query(query, {
            'id': doc_id,
            'limit': limit
        })
        return [record['related'] for record in results]
    
    def search_documents(
        self,
        properties: Dict[str, Any],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for documents with matching properties.
        
        Args:
            properties: Dictionary of property names and values to match
            limit: Maximum number of documents to return
            
        Returns:
            List of matching documents
        """
        conditions = " AND ".join(f"d.{k} = ${k}" for k in properties.keys())
        query = f"""
        MATCH (d:Document)
        WHERE {conditions}
        RETURN d
        LIMIT $limit
        """
        
        parameters = {**properties, 'limit': limit}
        results = self.execute_query(query, parameters)
        return [record['d'] for record in results]