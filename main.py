from modules.create_db_collection import build_db_collection
from modules.build_vector_db import create_vectored_texts
from modules.embeddings_model import embed_text

from pymilvus import MilvusClient
from datetime import datetime
now = datetime.now()

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model_action_recommendation_docs"

# milvus_client = build_db_collection(uri=uri, collection_name=collection_name)
# data = create_vectored_texts("Transformation Model")

# milvus_client.insert(collection_name=collection_name, data=data)

def insert_data():
    milvus_client = build_db_collection(uri=uri, collection_name=collection_name)
    data = create_vectored_texts("Files")

    milvus_client.insert(collection_name=collection_name, data=data)

milvus_client = MilvusClient(uri=uri)

def TGPS_retriever(question: str, timestamp: datetime= datetime.now()) -> str:
    # Convert the question into an embedding vector and perform a search
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            embed_text(question)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=2,  # Return top 2 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        filter=f'created_at < {int(timestamp.timestamp())}',
        output_fields=["source_id", "text", "created_at"],  # Return the source and text fields
    )

    # Process the search results to extract relevant information
    retrieved_lines_with_distances = [
        (res["entity"]["source_id"], res["entity"]["created_at"], res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    # Format the retrieved information into a readable context
    context = "\n".join(
        [line_with_distance[0] + ", " + str(datetime.fromtimestamp(line_with_distance[1])) + ": " + line_with_distance[2] for line_with_distance in retrieved_lines_with_distances]
    )
    return context

# if __name__ == "__main__":
#     # insert_data()

#     # # Example usage of the TGPS_retriever function
#     question = "What is Communicate to create readiness?"
#     print(TGPS_retriever(question=question, timestamp=datetime.now()))