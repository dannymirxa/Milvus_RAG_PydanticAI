from pymilvus import MilvusClient

# collection_name = "TGPS_transformation_model"

def build_milvus_client(uri: str, collection_name: str) -> MilvusClient:
    
    milvus_client = MilvusClient(uri= uri)

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=3072, # dimension size of text-embedding-3-large
        metric_type="IP", # L2 distance
        consistency_level="Strong",  # Strong consistency level
    )

    return milvus_client