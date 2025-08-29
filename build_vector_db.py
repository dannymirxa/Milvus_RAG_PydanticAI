from create_file_docs import create_file_data, create_docs_chunks_md, create_docs_chunks_pdf
# from embed_text import emb_text
from milvus_client import build_chat_client
from tqdm import tqdm

uri= "./milvus_tgps.db"
collection_name="TGPS_transformation_model_document"


from datetime import datetime
now = datetime.now()

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3")
# embeddings = model.encode(sentences)

def build_vector_db(file_path: str) -> None:

    milvus_client = build_chat_client(uri=uri, collection_name=collection_name)

    docs_chunks = create_docs_chunks_pdf(dir=file_path)

    data = []
    
    for index, doc in enumerate(tqdm(docs_chunks)):

        # page = str(doc['source_id']).split(".")[0]

        # start_datetime = datetime(2025, 8, int(page), 15, 30)

        vector =  model.encode(doc["text"])

        data.append(
            {
                "id": index, 
                "source_id": doc["source_id"], 
                "text": doc["text"],
                "vector": vector,
                # "created_at": int(start_datetime.timestamp())
                "created_at": int(now.timestamp())
            }
        )

    # print(data)

    milvus_client.insert(collection_name=collection_name, data=data)

build_vector_db("Files")


