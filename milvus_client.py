from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType

# This script sets up a Milvus client and creates a collection with specified fields.
# The collection is named "TGPS_transformation_model" and includes fields for storing
# text data and its corresponding vector embeddings.

def build_milvus_client(uri: str, collection_name: str) -> MilvusClient:
    # Initialize the Milvus client with the given URI
    milvus_client = MilvusClient(uri= uri)

    if milvus_client.has_collection(collection_name):
        milvus_client.drop_collection(collection_name)

    # Define the schema for the collection, specifying the fields and their data types
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )

    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048)
    schema.add_field(field_name="source_id", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=3072)

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
        params={ "nlist": 3072 }
    )

    # Create the index in the collection
    milvus_client.create_index(
        collection_name=collection_name,
        index_params=index_params
    )

    # Return the initialized Milvus client
    return milvus_client