import os
from dotenv import load_dotenv
from dataclasses import dataclass
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from openai import AsyncAzureOpenAI
from pymilvus import MilvusClient
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional
from annotated_types import MinLen
from pydantic_ai import Agent, ModelRetry, RunContext

from pymilvus import MilvusClient

load_dotenv()

async_client = AsyncAzureOpenAI(
    azure_endpoint = "https://llmcoechangemateopenai2.openai.azure.com/",
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21",
    azure_deployment='gpt-4o-dev'
)

OPENAI_MODEL = OpenAIModel(
    'gpt-4o',
    provider=OpenAIProvider(openai_client=async_client),
)

@dataclass
class Deps:
    openai: AsyncAzureOpenAI
    client: MilvusClient

class contextSuccess(BaseModel):
    context: Annotated[str, MinLen(1)] = Field(..., description='Context from vectore store')

agent = Agent(
    model=OPENAI_MODEL,
    output_type=contextSuccess,
    output_retries=3,
    model_settings={'temperature': 0.1}
)

@agent.tool
async def retriever(ctx: RunContext[Deps], query: str) -> str:
    embedding = await ctx.deps.openai.embeddings.create(input=query, model='text-embedding-3-large')
    embedding = embedding.data[0].embedding

    search_res = ctx.deps.client.search(
        collection_name="TGPS_transformation_model",
        data=[
            embedding
        ],  
        limit=2,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["source", "text"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["source"], res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    ctx = "\n".join(
        [line_with_distance[0] + ": " + line_with_distance[1] for line_with_distance in retrieved_lines_with_distances]
    )
    return ctx

@agent.system_prompt
def system_prompt(ctx: RunContext[Deps]) -> str:
    return f"""
    You are an AI assistant. Your main goal is to answer the user's questions accurately and comprehensively.
    To achieve this, you must use the `retriever` tool to find relevant information.

    Here's how you should operate:
    1. When the user asks a question, identify the key terms or concepts in their query.
    2. Use these key terms as the `query` for the `retriever` tool to fetch relevant context.
    3. Once you receive the context from the `retriever` tool, carefully read and understand it.
    4. Formulate your answer to the user's question *solely based on the information provided in the retrieved context*.
    5. If the retrieved context does not contain enough information to answer the question, state that you cannot answer based on the available information.
    6. Do not make up information or use external knowledge. Always rely on the `retriever` tool's output.
    """

async def main(request: str):
    openai_client = AsyncAzureOpenAI(        
        api_version="2024-12-01-preview",
        azure_endpoint=os.getenv("AZURE_OPEN_AI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )
    deps = Deps(openai=openai_client, client=MilvusClient(uri="./milvus_tgps.db"))
    response = await agent.run(user_prompt=request, deps=deps)

    return response, response.usage()

import asyncio

if __name__ == "__main__":
    response, total_tokens = asyncio.run(main("what needs to be doen to manage rumors?"))

    print(response)
    print(total_tokens)
