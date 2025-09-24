from openai import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from neo4j import GraphDatabase

# -------------------------------
# 1. Set up OpenAI
# -------------------------------
OPENAI_API_KEY = "<your-openai-api-key>"
client = OpenAI(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# -------------------------------
# 2. Set up Neo4j AuraDB
# -------------------------------
NEO4J_URI = "neo4j+s://b8cef351.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "VwOQPOfB5YX8KfrHWfDOuo9pMScCDVK0omeItsni7tg"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------------
# 3. Example: LLM generates Knowledge Graph
# -------------------------------
# Here, we're simulating KG creation; in practice, you'd call an LLM to generate nodes & relations
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
# 4. Store KG in Neo4j
# -------------------------------
with driver.session() as session:
    # Create nodes
    for node in kg_nodes:
        session.run(
            "MERGE (n:Entity {name: $name, type: $type})",
            name=node["name"],
            type=node["type"]
        )
    # Create relationships
    for edge in kg_edges:
        session.run(
            """
            MATCH (a:Entity {name: $from_name})
            MATCH (b:Entity {name: $to_name})
            MERGE (a)-[r:RELATION {type: $relation}]->(b)
            """,
            from_name=edge["from"],
            to_name=edge["to"],
            relation=edge["relation"]
        )

# -------------------------------
# 5. Generate embeddings for nodes
# -------------------------------
texts = [node["name"] + " " + node["type"] for node in kg_nodes]
text_embeddings = embeddings.embed_documents(texts)
text_embedding_pairs = list(zip(texts, text_embeddings))

# -------------------------------
# 6. Store embeddings using Neo4jVector
# -------------------------------
vectorstore = Neo4jVector.from_embeddings(
    text_embedding_pairs,
    embeddings,
    url=NEO4J_URI,
    username=NEO4J_USER,
    password=NEO4J_PASSWORD,
    node_label="Entity",                 # Same label as your KG nodes
    text_node_property="name",           # Property to store text
    embedding_node_property="embedding"  # Property to store embedding
)

# -------------------------------
# 7. Example: Semantic search
# -------------------------------
query = "Which database does LangChain integrate with?"
results = vectorstore.similarity_search(query, k=3)

print("Top matches:")
for r in results:
    print(r["content"], r["score"])

