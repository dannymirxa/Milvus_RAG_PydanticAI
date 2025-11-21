from pathlib import Path
import sys
from datetime import datetime
from pymilvus import MilvusClient

# Ensure the project root is on sys.path so package imports like `modules.*` work
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from modules.embeddings_model import embed_text

def retriever(milvus_client: MilvusClient, collection_name: str, question: str, timestamp: datetime = datetime.now()) -> str:
    # Convert the question into an embedding vector and perform a search
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            embed_text(question)
        ],  # Use the `embed_text` function to convert the question to an embedding vector
        limit=1,  # Return top 2 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        filter=f'created_at < {int(timestamp.timestamp())}',
        output_fields=["source_id", "text", "created_at"],  # Return the source and text fields
    )

    # Process the search results to extract relevant information
    retrieved_lines_with_distances = [
        (res["entity"]["source_id"], res["entity"]["created_at"], res["entity"]["text"]) for res in search_res[0]
    ]

    # Format the retrieved information into a readable context
    context = "\n".join(
        [line_with_distance[0] + ", " + str(datetime.fromtimestamp(line_with_distance[1])) + ": " + line_with_distance[2] for line_with_distance in retrieved_lines_with_distances]
    )
    return context


question = "What is Communicate to create readiness?"
print(retriever(milvus_client=MilvusClient(uri="./milvus_tgps.db"), collection_name="TGPS_transformation_model_action_recommendation_docs", question=question, timestamp=datetime.now()))