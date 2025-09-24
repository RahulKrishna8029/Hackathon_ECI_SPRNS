# SPRNS - Smart Retrieval & Knowledge System

SPRNS is a Retrieval-Augmented Generation (RAG) system that combines semantic search with intelligent answer generation to provide accurate, context-aware responses from a knowledge base.

## Features

- **Semantic Search**: Uses advanced embedding models to find relevant documents
- **Context-Aware Generation**: Generates answers using retrieved context
- **Source Attribution**: Shows which documents were used to generate answers
- **Graph Database Integration**: Leverages Neo4j for knowledge graph storage
- **Interactive Dashboard**: User-friendly Streamlit interface
- **Modular Architecture**: Easily extensible and configurable

## Architecture

```
SPRNS/
├── Dashboard/
│   └── sprns_ui.py          # Streamlit web interface
├── retrieval/
│   ├── core/
│   │   ├── query_processor.py    # Query embedding and processing
│   │   └── answer_generator.py   # Answer generation and reranking
│   ├── utils/
│   │   └── neo4j_connector.py    # Neo4j database interface
│   └── retrieval_service.py      # Main service orchestrator
├── config.py                     # Configuration settings
├── run_dashboard.py              # Dashboard launcher
└── test_integration.py           # Integration tests
```

## Quick Start

### 1. Install Dependencies

```bash
pip install streamlit torch transformers sentence-transformers neo4j
```

### 2. Test the Integration

```bash
python test_integration.py
```

### 3. Run the Dashboard

```bash
python run_dashboard.py
```

Or manually:

```bash
streamlit run Dashboard/sprns_ui.py
```

The dashboard will be available at `http://localhost:8501`

## Configuration

### Environment Variables

You can configure SPRNS using environment variables:

```bash
# Neo4j Configuration
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your_password"

# Model Configuration
export EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
export GENERATION_MODEL="google/flan-t5-base"

# Dashboard Configuration
export DASHBOARD_PORT="8501"
export DASHBOARD_HOST="localhost"
```

### Configuration File

Modify `config.py` to change default settings:

```python
class Config:
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USERNAME = "neo4j"
    NEO4J_PASSWORD = "password"
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    GENERATION_MODEL = "google/flan-t5-base"
```

## Usage

### Dashboard Interface

1. **Ask Questions**: Type your question in the chat input
2. **View Answers**: Get AI-generated responses based on retrieved documents
3. **Check Sources**: Expand the "Sources" section to see supporting documents
4. **System Status**: Monitor connection status in the sidebar

### Programmatic Usage

```python
from retrieval.retrieval_service import RetrievalService

# Initialize service
service = RetrievalService()

# Process a query
result = service.process_query("What is machine learning?")

print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])}")

# Clean up
service.close()
```

## Components

### Query Processor
- Converts text queries into embeddings
- Classifies query intent
- Extracts parameters and filters

### Answer Generator
- Reranks retrieved documents by relevance
- Generates contextual answers using language models
- Provides source attribution

### Neo4j Connector
- Manages graph database connections
- Executes Cypher queries
- Handles document storage and retrieval

### Retrieval Service
- Orchestrates the entire RAG pipeline
- Handles error cases and fallbacks
- Provides a unified interface

## Development

### Adding New Models

1. Update model names in `config.py`
2. Ensure compatibility with the existing interfaces
3. Test with `test_integration.py`

### Extending Functionality

1. Add new methods to the appropriate core modules
2. Update the `RetrievalService` to use new functionality
3. Modify the dashboard UI if needed

### Testing

Run the integration tests:

```bash
python test_integration.py
```

## Troubleshooting

### Neo4j Connection Issues

- Ensure Neo4j is running on the specified URI
- Check username and password
- Verify network connectivity
- The system will use mock data if Neo4j is unavailable

### Model Loading Issues

- Ensure you have sufficient memory for the models
- Check internet connectivity for model downloads
- Consider using smaller models for testing

### Dashboard Issues

- Ensure Streamlit is installed: `pip install streamlit`
- Check that the port is not in use
- Verify all dependencies are installed

## License

This project is part of the Hackathon ECI SPRNS initiative.