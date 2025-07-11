from create_file_docs import create_file_data, create_docs_chunks
# from embed_text import emb_text
from milvus_client import build_chat_client
from tqdm import tqdm

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model_timestamped"


from datetime import datetime
now = datetime.now()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
# embeddings = model.encode(sentences)

def build_vector_db() -> None:

    milvus_client = build_chat_client(uri=uri, collection_name=collection_name)

    docs_chunks = create_docs_chunks(dir="Transformation Model")

    data = []
    for index, doc in enumerate(tqdm(docs_chunks)):

        page = str(doc['source_id']).split(".")[0]

        start_datetime = datetime(2025, 7, int(page), 15, 30)

        vector =  model.encode(doc["text"])

        data.append(
            {
                "id": index, 
                "source_id": doc["source_id"], 
                "text": doc["text"], 
                "vector": vector,
                "created_at": int(start_datetime.timestamp())
            }
        )

    # print(data)

    milvus_client.insert(collection_name=collection_name, data=data)

build_vector_db()


