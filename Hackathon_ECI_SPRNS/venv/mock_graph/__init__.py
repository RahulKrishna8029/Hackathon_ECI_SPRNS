"""
Mock Graph Database Package for SPRNS System.
Provides Neo4j simulation with customer data for local testing.
"""

from .mock_neo4j import MockNeo4jConnector

__all__ = ['MockNeo4jConnector']