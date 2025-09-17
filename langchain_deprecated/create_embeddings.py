import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv(".env")

embeddings = AzureOpenAIEmbeddings(
    model="text-embedding-3-large",
    # dimensions: Optional[int] = None, # Can specify dimensions with new text-embedding-3 models
    azure_endpoint=os.getenv("AZURE_OPEN_AI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_version="2024-12-01-preview"
)