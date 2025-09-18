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
from pydantic_ai import Agent, ModelRetry, RunContext

from dotenv import load_dotenv
load_dotenv('.env')

from model import OPENAI_MODEL
from modules.embeddings_model import embed_text
from tools.retriever import retriever

@dataclass
class Deps:
    client: MilvusClient
    collection_name: str

class actionSuccess(BaseModel):
    context: Annotated[str, MinLen(1)] = Field(..., description='Context from vectore store')

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
        collection_name=ctx.deps.collection_name, 
        question=query
    )

@agent.system_prompt
def system_prompt(ctx: RunContext[Deps]) -> str:
    return f"""
    You are an AI assistant. Use only the `tgps_retriever` tool to fetch context for user queries.
    For each user question:
    1. Decide if `tgps_retriever` is applicable (if unsure, use it).
    2. Call `tgps_retriever` with key terms from the user's query.
    3. Base your response exclusively on the retrieved context.
    4. If the retrieved context is insufficient, state that you cannot answer from the available information.
    Do not invent facts or use external knowledge beyond the retriever output.
    """

def main(request: str):
    deps = Deps(client=MilvusClient(uri="./milvus_tgps.db"), collection_name="TGPS_transformation_model_action_recommendation_docs")
    response = agent.run_sync(user_prompt=request, deps=deps)

    print(response.output)

if __name__ == "__main__":
    main("What is Communicate to create readiness?")
