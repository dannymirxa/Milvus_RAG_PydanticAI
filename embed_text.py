import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = "https://llmcoechangemateopenai2.openai.azure.com/"
deployment = "text-embedding-3-large"

api_version = "2024-02-01"

openai_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint=endpoint,
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

def emb_text(text: str):
    return (
        openai_client.embeddings.create(input=text, model=deployment)
        .data[0]
        .embedding
    )