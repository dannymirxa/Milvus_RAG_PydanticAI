from pymilvus import connections, Collection, MilvusClient

def check_collection():
    # Connect to Milvus Lite (SQLite-based)
    connections.connect(alias="default", uri="milvus_tgps.db")

    # client = MilvusClient("./milvus_demo.db")

    # Load the collection
    # collection = Collection("TGPS_transformation_chat")
    collection = Collection("TGPS_transformation_model_document")

    # Load data into memory
    collection.load()

    # Retrieve all data (you can limit or filter as needed)
    results = collection.query(
        expr="id == 103",  # No filter, fetch all
        filter="id == 103",
        output_fields=["id", "source_id", "page", "text", "created_at"]
    )

    # Print results
    for item in results:
        print(item)

from datetime import datetime
from datetime import datetime
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")

async def chat_retriever(query: str, timestamp: datetime= datetime.now()) -> str:
    # embedding = await ctx.deps.openai.embeddings.create(input=query, model='text-embedding-3-large')
    # embedding = embedding.data[0].embedding

    search_res = MilvusClient(uri="./milvus_tgps.db").search(
        collection_name="TGPS_transformation_chat",
        data=[
            model.encode(query)
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

if __name__ == "__main__":

    # import asyncio

    # result = asyncio.run(chat_retriever("How"))

    # print(result)

    check_collection()