from create_embeddings import embeddings
from langchain_milvus import Milvus

# The easiest way is to use Milvus Lite where everything is stored in a local file.
# If you have a Milvus server you can use the server URI such as "http://localhost:19530".
URI = "langchain/milvus_demo.db"

vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": URI},
    collection_name="TGPS_transformation_model",
)

query = "What is benefits realization?"
results = vectorstore.similarity_search(query, k=1)

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

