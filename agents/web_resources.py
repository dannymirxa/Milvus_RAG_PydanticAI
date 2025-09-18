from pathlib import Path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass
from pymilvus import MilvusClient
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional, List
from annotated_types import MinLen
from pydantic_ai import Agent, RunContext

from dotenv import load_dotenv
load_dotenv('.env')

from model import OPENAI_MODEL
from tools.ddgs_search import ddgs_search

@dataclass
class Deps:
    client: MilvusClient

class webSuccess(BaseModel):
    webAnswer: Annotated[str, MinLen(1)] = Field(..., description='Answer from web')
    references: List[str] = Field(..., description='List of reference URLs from web search')

class invalidRequest(BaseModel):
    error_message: str

webResponse: TypeAlias = Union[webSuccess, invalidRequest]

agent = Agent(
    model=OPENAI_MODEL,
    output_type=webResponse,
    output_retries=3,
    model_settings={'temperature': 0.1}
)

@agent.tool
def web_result(ctx: RunContext[Deps], query: str) -> str:
    return ddgs_search(query=query, max_results=3)


@agent.system_prompt
def system_prompt(ctx: RunContext[Deps]) -> str:
    return f"""
    You are an AI assistant restricted to using a single web search tool: `web_result`.
    For every user question:
    1. Use ONLY the `web_result` tool to search the web for relevant information. Do NOT call any other tools or retrievers.
    2. Produce a detailed, evidence-backed answer based only on the web search results and place that text exactly into the `webAnswer` field of the agent's final output.
       - The answer should be thorough: provide a clear short summary (1-2 sentences), followed by a detailed explanation (at least 3-5 sentences) that covers causes, implications, examples, or steps as applicable to the question.
       - When helpful, include a short "Key points" bullet-style list inside `webAnswer` summarizing the most important facts (use plain text bullets like "- ").
    3. Extract and return the source URLs used to form the answer as a JSON list in the `references` field. Each item must be a full URL string. The `references` list must include every distinct URL you relied on.
    4. Where specific claims are made,s ensure they are supported by at least one URL in `references`. If multiple sources support a claim, prefer the most authoritative and include multiple references when relevant.
    5. If the web search results are insufficient to answer the question, set `webAnswer` to a short explicit note (for example: "no relevant web results found") and set `references` to an empty list ([]).
    6. Do not invent facts beyond what is supported by the retrieved web results. When uncertain about a claim, state the uncertainty and cite the sources that led to that assessment.
    7. Keep the answer focused and factual; avoid unrelated background. The only output fields the agent will return are `webAnswer` (string) and `references` (list of strings).
    """

# def main(request: str):
#     deps = Deps(client=MilvusClient(uri="./milvus_tgps.db"))
#     response = agent.run_sync(user_prompt=request, deps=deps)

#     print(response.output)

# if __name__ == "__main__":
#     main("How to communicate to create readiness?")
