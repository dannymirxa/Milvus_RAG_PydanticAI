from create_file_docs import create_file_data
from embed_text import emb_text
from milvus_client import build_milvus_client
from tqdm import tqdm

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model"

def build_vector_db() -> None:

    milvus_client = build_milvus_client(uri=uri, collection_name=collection_name)

    files = create_file_data(dir="Transformation Model")

    data = []

    for i, file_data in enumerate(tqdm(files, desc="Creating embeddings")):
        data.append({"id": i, "source": file_data["source"], "text": file_data["text"], "vector": emb_text(file_data["text"])})

    # print(data)

    milvus_client.insert(collection_name=collection_name, data=data)

build_vector_db()


