from modules.create_file_docs import create_file_data, create_docs_chunks_md, create_docs_chunks_pdf
from modules.embeddings_model import embed_text
from tqdm import tqdm

from datetime import datetime
now = datetime.now()

# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer("BAAI/bge-m3")
# embeddings = model.encode(sentences)

def create_vectored_texts(file_path: str) -> None:
    docs_chunks = create_docs_chunks_md(dir=file_path)
    data = []
    
    for index, doc in enumerate(tqdm(docs_chunks)):
        # page = str(doc['source_id']).split(".")[0]
        # start_datetime = datetime(2025, 8, int(page), 15, 30)
        vector =  embed_text(doc["text"])
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
    return data
    


