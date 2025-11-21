# Milvus RAG PydanticAI

## Overview

This project implements Retrieval-Augmented Generation (RAG) using a local Milvus Lite vector store, Azure OpenAI embeddings, and a LangGraph pipeline with PydanticAI agents. Data is embedded with the Azure OpenAI deployment "text-embedding-3-large" and stored in a Milvus collection with 3072‑dim vectors. The pipeline retrieves context from the vector store, performs a focused web search, and then compiles a final answer.

Key components:
- [modules/embeddings_model.py](modules/embeddings_model.py) — [python.def embed_text()](modules/embeddings_model.py:18) generates 3072‑dim embeddings using Azure OpenAI.
- [modules/create_db_collection.py](modules/create_db_collection.py) — [python.def build_db_collection()](modules/create_db_collection.py:3) creates a Milvus collection and builds an index.
- [modules/vectorize.py](modules/vectorize.py) — [python.def create_vectored_texts()](modules/vectorize.py:12) chunks markdown files and prepares records with embeddings and timestamps.
- [tools/retriever.py](tools/retriever.py) — [python.def retriever()](tools/retriever.py:13) performs IP metric search with an optional timestamp filter.
- [agents/action_recommendation.py](agents/action_recommendation.py) — [python.def tgps_retriever()](agents/action_recommendation.py:41) and [python.def docs_retriever()](agents/action_recommendation.py:49) feed the agent returning [python.class actionSuccess](agents/action_recommendation.py:24).
- [agents/web_resources.py](agents/web_resources.py) — [python.def web_result()](agents/web_resources.py:41) leverages DuckDuckGo search to gather references.
- [model.py](model.py) — [python.def OPENAI_MODEL](model.py:16) configures the chat model provider for agents.
- [main.py](main.py) — orchestrates the LangGraph via [python.def create_graph()](main.py:93) and runs it with [python.def main()](main.py:130).

## Prerequisites

- Python 3.10+ recommended.
- Create a `.venv` and activate it.
- Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   See [requirements.txt](requirements.txt) for the full list, including `pymilvus`, `milvus-lite`, `openai`, `python-dotenv`, `pydantic-ai-slim`, `langgraph`, and `ddgs`.

- Azure OpenAI access for embeddings is required.

## Configure environment

Set the following in [.env](.env):

- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- LANGSMITH_PROJECT (optional, tracing/observability)
- LANGSMITH_API_KEY (optional)

The embeddings client is configured in [python.def embed_text()](modules/embeddings_model.py:18) using deployment `text-embedding-3-large`.

## Build the vector database

This project uses a local Milvus client with URI `./milvus_tgps.db`. To create the collection and insert embedded records:

1. Ensure your source markdown files exist under a directory named "Files" in the project root. The chunking and embedding pipeline reads markdown via [python.def create_vectored_texts()](modules/vectorize.py:12).
2. Run the one-time insert script:

   ```bash
   python -c "from insert_data import insert_data; insert_data()"
   ```

   This calls [python.def insert_data()](insert_data.py:20), which:
   - Creates (or recreates) the collection via [python.def build_db_collection()](modules/create_db_collection.py:3) with:
     - Fields: `id` (INT64, primary), `text` (VARCHAR, 3072), `source_id` (VARCHAR, 256), `references` (VARCHAR, 3072), `vector` (FLOAT_VECTOR, dim=3072), `created_at` (INT64).
     - Metric type: Inner Product (IP), index type FLAT on `vector`.
   - Generates records from markdown using [python.def create_vectored_texts()](modules/vectorize.py:12).
   - Inserts data into the collection `TGPS_transformation_model_action_recommendation_docs`.

Note: Re-running [python.def build_db_collection()](modules/create_db_collection.py:3) drops and recreates the collection.

## Collections

- Primary collection used by examples: `TGPS_transformation_model_action_recommendation_docs`.
- The action recommendation agent also references `TGPS_transformation_model_action_recommendation` via [python.def tgps_retriever()](agents/action_recommendation.py:41). Populate it similarly if required.

## Usage

- Run the LangGraph pipeline:

   ```bash
   python main.py
   ```

   [main.py](main.py) streams node completion events after [python.def create_graph()](main.py:93) compiles the graph and [python.def main()](main.py:130) drives execution with an initial question.

- Generate a graph PNG (optional):

   ```bash
   python -c "import main; main.create_graph_image()"
   ```

   This saves [graph.png](graph.png) using [python.def create_graph_image()](main.py:120).

- Retrieve directly from Milvus (without agents):

   ```python
   from pymilvus import MilvusClient
   from tools.retriever import retriever

   client = MilvusClient(uri="./milvus_tgps.db")
   ctx = retriever(milvus_client=client, collection_name="TGPS_transformation_model_action_recommendation_docs", question="What are the importance of business models?")
   print(ctx)
   ```

   This uses [python.def retriever()](tools/retriever.py:13), which:
   - Embeds the query via [python.def embed_text()](modules/embeddings_model.py:18).
   - Searches with `limit=1`, metric IP, and `filter='created_at < now'`.
   - Returns a newline-joined context of `source_id`, timestamp, and text.

- Example using the TGPS helper:

   ```python
   from insert_data import TGPS_retriever
   from datetime import datetime

   print(TGPS_retriever(question="What are the importance of business models?", timestamp=datetime.now()))
   ```

   See [python.def TGPS_retriever()](insert_data.py:28) (uses `limit=2` and the same timestamp filter).

## Agent architecture

- Action Recommendation Agent: [agents/action_recommendation.py](agents/action_recommendation.py)
  - Tools: [python.def tgps_retriever()](agents/action_recommendation.py:41), [python.def docs_retriever()](agents/action_recommendation.py:49)
  - Output model: [python.class actionSuccess](agents/action_recommendation.py:24) with `recommendationAnswer` and `docsAnswer`.
- Web Resources Agent: [agents/web_resources.py](agents/web_resources.py)
  - Tool: [python.def web_result()](agents/web_resources.py:41)
  - Returns `webAnswer` and `references`.
- Compilation Agent: [agents/summary.py](agents/summary.py)
  - Combines agent outputs and returns a final compiled response consumed in [python.def compilation_node()](main.py:74).

## Notes

- Milvus runs locally via [pymilvus](https://pymilvus.readthedocs.io/) and `milvus-lite`; the client is initialized with a file URI (`./milvus_tgps.db`) in multiple modules including [main.py](main.py) and [tools/retriever.py](tools/retriever.py).
- The collection schema dimension (3072) matches Azure OpenAI `text-embedding-3-large`. Changing the embedding model requires updating [python.def build_db_collection()](modules/create_db_collection.py:3) and re-inserting data.
- Keep secrets out of version control. Prefer environment variables via [.env](.env) and never commit real API keys.

## License

This project is licensed under the MIT License.