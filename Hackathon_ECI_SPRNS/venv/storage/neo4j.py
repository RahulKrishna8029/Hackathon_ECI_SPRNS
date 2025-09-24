# -------------------------------
# imports
# -------------------------------
import os
import numpy as np
from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from neo4j import GraphDatabase
from src.proj.projection import ProjectionModelPCA   # your projection module

# -------------------------------
# 1. Load API keys (env vars instead of hardcoding)
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")       # e.g. neo4j+s://<your-id>.databases.neo4j.io
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

if not all([OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD]):
    raise ValueError("Missing required environment variables.")

# Clients
client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------------
# 2. Define KG
# -------------------------------
kg_nodes = [
    {"name": "Neo4j", "type": "Database"},
    {"name": "LangChain", "type": "Framework"},
    {"name": "OpenAI", "type": "AI Provider"}
]

kg_edges = [
    {"from": "LangChain", "to": "Neo4j", "relation": "integrates_with"},
    {"from": "LangChain", "to": "OpenAI", "relation": "uses"}
]

# -------------------------------
# 3. Store KG in Neo4j
# -------------------------------
with driver.session() as session:
    for node in kg_nodes:
        session.run(
            "MERGE (n:Entity {name: $name}) "
            "SET n.type = $type",
            name=node["name"],
            type=node["type"]
        )
    for edge in kg_edges:
        session.run(
            """
            MATCH (a:Entity {name: $from_name})
            MATCH (b:Entity {name: $to_name})
            MERGE (a)-[:RELATION {type: $relation}]->(b)
            """,
            from_name=edge["from"],
            to_name=edge["to"],
            relation=edge["relation"]
        )

# -------------------------------
# 4. Generate + project embeddings
# -------------------------------
texts = [node["name"] + " " + node["type"] for node in kg_nodes]
raw_embeddings = embeddings.embed_documents(texts)

# PCA projection → dimension reduced to 32
d = len(raw_embeddings[0])
projector = ProjectionModelPCA(d=d, k=32)
projector.fit_initial(np.array(raw_embeddings))
projected_embeddings = [projector.project(np.array(e)).tolist() for e in raw_embeddings]

# -------------------------------
# 5. Attach embeddings to KG nodes
# -------------------------------
with driver.session() as session:
    for node, proj_vec in zip(kg_nodes, projected_embeddings):
        session.run(
            """
            MATCH (n:Entity {name: $name})
            SET n.proj_embedding = $embedding
            """,
            name=node["name"],
            embedding=proj_vec
        )

# -------------------------------
# 6. Create vector index (once only)
# -------------------------------
with driver.session() as session:
    session.run("""
        CREATE VECTOR INDEX entity_proj_index IF NOT EXISTS
        FOR (n:Entity) ON (n.proj_embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 32,
                `vector.similarity_function`: 'cosine'
            }
        }
    """)

print("✅ KG + embeddings stored in Neo4j Aura. You can now query via the UI.")
