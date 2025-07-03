import os
import glob

from langchain_milvus import BM25BuiltInFunction, Milvus
from create_documents import create_documents
from create_embeddings import embeddings

directory_path = "Transformation Model"
markdown_files = glob.glob(os.path.join(directory_path, "*.md"))

docs = create_documents(markdown_files)

vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    connection_args={
        "uri": "langchain/milvus_demo.db",
    },
    collection_name="TGPS_transformation_model",
    consistency_level="Strong",
    drop_old=False,  # Drop the old Milvus collection if it exists
)

# query = "What is benefits realization?"
# results = vectorstore.similarity_search(query, k=1)

# for res in results:
#     print(f"* {res.page_content} [{res.metadata}]")