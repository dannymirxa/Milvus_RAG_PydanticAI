from pymilvus import MilvusClient, DataType
from create_file_docs import create_file_data, create_docs_chunks
# from embed_text import emb_text
from milvus_client import build_chat_client
from tqdm import tqdm

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_chat"

from datetime import datetime

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
# embeddings = model.encode(sentences)

def build_chat_client(uri: str, collection_name: str) -> MilvusClient:
    # Initialize the Milvus client with the given URI
    milvus_client = MilvusClient(uri= uri)

    # if milvus_client.has_collection(collection_name):
    #     milvus_client.drop_collection(collection_name)

    # Define the schema for the collection, specifying the fields and their data types
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="created_at", datatype=DataType.INT64)

    # Create the collection with the defined schema
    milvus_client.create_collection(
        collection_name=collection_name,
        schema=schema,
        metric_type="IP",
        consistency_level="Strong",
    )

    # Prepare index parameters for the vector field
    index_params = MilvusClient.prepare_index_params()

    # Add an index to the vector field to optimize search operations
    index_params.add_index(
        field_name="vector",
        metric_type="IP",
        index_type="FLAT",
        index_name="vector_index",
        params={ "nlist": 1024 }
    )

    # Create the index in the collection
    milvus_client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )

    # Return the initialized Milvus client
    return milvus_client

def insert_chat_into_vector_db(request: str, output: str) -> None:
    milvus_client = build_chat_client(uri=uri, collection_name=collection_name)

    chat =  f"""
            User: {request}

            Assistant: {output}
            """

    data = [{
                "text": chat, 
                "vector": model.encode(chat),
                "created_at": int(datetime.now().timestamp())
            }]

    milvus_client.insert(collection_name=collection_name, data=data)

# build_chat_client(uri=uri, collection_name=collection_name)

# client=MilvusClient(uri="./milvus_tgps.db")
# query = "what needs to be done to manage rumors?"
# timestamp = datetime(2025, 7, 12, 15, 30)
# search_res = client.search(
#                 collection_name=collection_name,
#                 data=[
#                     model.encode(query)
#                 ],  
#                 limit=2,  # Return top 3 results
#                 search_params={"metric_type": "IP", "params": {}},  # Inner product distance
#                 filter=f'created_at < {int(timestamp.timestamp())}',
#                 output_fields=["created_at", "text"],  # Return the text field
#             )

# retrieved_lines_with_distances = [
#     (datetime.fromtimestamp(res["entity"]["created_at"]), res["entity"]["text"], res["distance"]) for res in search_res[0]
# ]

# ctx = "\n".join(
#         [str(line_with_distance[0]) + ": " + line_with_distance[1] for line_with_distance in retrieved_lines_with_distances]
#     )

# print(ctx)