from pymilvus import MilvusClient
# from embed_text import emb_text

from embed_text import emb_text

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model"

milvus_client = MilvusClient(uri=uri)


def TGPS_retriever(question: str) -> str:
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(question)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=2,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["source", "text"],  # Return the text field
    )

    import json

    retrieved_lines_with_distances = [
        (res["entity"]["source"], res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]
    # print(json.dumps(retrieved_lines_with_distances, indent=4))

    context = "\n".join(
        [line_with_distance[0] + ": " + line_with_distance[1] for line_with_distance in retrieved_lines_with_distances]
    )
    return context


question = "What is the main factor of Fear and Frustration?"

print(TGPS_retriever(question=question))