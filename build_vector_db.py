from create_file_docs import create_file_data, create_docs_chunks
from embed_text import emb_text
from milvus_client import build_milvus_client
from tqdm import tqdm

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model"

def build_vector_db() -> None:

    milvus_client = build_milvus_client(uri=uri, collection_name=collection_name)

    docs_chunks = create_docs_chunks(dir="Transformation Model")

    data = []
    for index, doc in enumerate(tqdm(docs_chunks)):
        vector = emb_text(doc["text"])
        doc["vector"] = vector

        data.append(
            {
            "id": index, 
            "source_id": doc["source_id"], 
            "text": doc["text"], 
            "vector": emb_text(doc["text"])
            }
        )

    # print(data)

    milvus_client.insert(collection_name=collection_name, data=data)

build_vector_db()


