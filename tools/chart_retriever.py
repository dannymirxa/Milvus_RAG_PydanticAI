from datetime import datetime
from pymilvus import MilvusClient
from modules.embeddings_model import embed_text



def chat_retriever(milvus_client: MilvusClient, query: str, timestamp: datetime= datetime.now()) -> str:
    # embedding = await ctx.deps.openai.embeddings.create(input=query, model='text-embedding-3-large')
    # embedding = embedding.data[0].embedding

    search_res = milvus_client.search(
        collection_name="chat",
        data=[
            embed_text(query)
        ],  
        limit=2,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        filter=f'created_at < {int(timestamp.timestamp())}',
        output_fields=["text", "created_at"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], str(datetime.fromtimestamp(res["entity"]["created_at"])), res["distance"]) for res in search_res[0]
    ]

    context = "\n".join(
        ["time: " + line_with_distance[1] + "\n" + line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )
    return context