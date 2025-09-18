from pathlib import Path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass
from pymilvus import MilvusClient
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional
from annotated_types import MinLen
from pydantic_ai import Agent, RunContext

from dotenv import load_dotenv
load_dotenv('.env')

from model import OPENAI_MODEL
from tools.retriever import retriever

@dataclass
class Deps:
    client: MilvusClient

class actionSuccess(BaseModel):
    recommendationAnswer: Annotated[str, MinLen(1)] = Field(..., description='Answer from recommendation vector store')
    docsAnswer: Annotated[str, MinLen(1)] = Field(..., description='Answer from documents vector store')

class invalidRequest(BaseModel):
    error_message: str

actionResponse: TypeAlias = Union[actionSuccess, invalidRequest]

agent = Agent(
    model=OPENAI_MODEL,
    output_type=actionResponse,
    output_retries=3,
    model_settings={'temperature': 0.1}
)

@agent.tool
def tgps_retriever(ctx: RunContext[Deps], query: str) -> str:
    return retriever(
        milvus_client=ctx.deps.client, 
        collection_name="TGPS_transformation_model_action_recommendation", 
        question=query
    )

@agent.tool
def docs_retriever(ctx: RunContext[Deps], query: str) -> str:
    return retriever(
        milvus_client=ctx.deps.client, 
        collection_name="TGPS_transformation_model_action_recommendation_docs", 
        question=query
    )

@agent.system_prompt
def system_prompt(ctx: RunContext[Deps]) -> str:
    return f"""
    You are an AI assistant. Use the `tgps_retriever` and `docs_retriever` tools to fetch context for user queries.
    For each user question:
    1. Decide whether the question needs information from the recommendation vector store (`tgps_retriever`), the documents vector store (`docs_retriever`), or both. If unsure, call both.
    2. When relevant, call `tgps_retriever` with key terms from the user's query to retrieve recommendation-context answers. The answer returned from `tgps_retriever` must be placed in the final response field `recommendationAnswer`.
    3. When relevant, call `docs_retriever` with key terms from the user's query to retrieve document-based answers. The answer returned from `docs_retriever` must be placed in the final response field `docsAnswer`.
    4. Combine the retrieved contexts only when both are applicable; otherwise, base your response solely on the single retriever's output. Even when combining, ensure the `recommendationAnswer` contains the `tgps_retriever` result and `docsAnswer` contains the `docs_retriever` result.
    5. If retrieved context is insufficient to answer, state that you cannot answer from the available information and populate the missing field(s) with a short explicit note (e.g., "no relevant recommendation context found" or "no relevant document context found").
    Do not invent facts or use external knowledge beyond the retriever outputs.
    """

# def main(request: str):
#     deps = Deps(client=MilvusClient(uri="./milvus_tgps.db"))
#     response = agent.run_sync(user_prompt=request, deps=deps)

#     print(response.output)

# if __name__ == "__main__":
#     main("How to communicate to create readiness?")
