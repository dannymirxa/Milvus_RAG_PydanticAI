from pymilvus import MilvusClient
# from embed_text import emb_text

# This script demonstrates how to use the Milvus client to perform a search
# on the "TGPS_transformation_model" collection. It retrieves text data based
# on a given question by converting the question into an embedding vector.

from datetime import datetime
now = datetime.now()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model_timestamped"

# Initialize the Milvus client with the specified URI
milvus_client = MilvusClient(uri=uri)

def TGPS_retriever(question: str, timestamp: datetime= datetime.now()) -> str:
    # Convert the question into an embedding vector and perform a search
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            model.encode(question)
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

# Example usage of the TGPS_retriever function
question = "What is fear?"
print(TGPS_retriever(question=question, timestamp=datetime.now()))