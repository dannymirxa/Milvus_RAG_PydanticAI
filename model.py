import os
from openai import AsyncAzureOpenAI
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from dotenv import load_dotenv
load_dotenv('.env')

async_client = AsyncAzureOpenAI(
    azure_endpoint = "https://llmcoechangemateopenai2.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21",
    azure_deployment='gpt-4o-dev'
)

OPENAI_MODEL = OpenAIChatModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=async_client),
)
