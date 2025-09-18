from pathlib import Path
import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataclasses import dataclass
from pymilvus import MilvusClient
from pydantic import BaseModel, Field
from typing_extensions import Annotated, TypeAlias, Union, Optional
from typing import List
from annotated_types import MinLen
from pydantic_ai import Agent, RunContext

from dotenv import load_dotenv
load_dotenv('.env')

from model import OPENAI_MODEL

@dataclass
class Deps:
    action_agent_response_recommendationAnswer: str
    action_agent_response_docsAnswer: str
    web_agent_response: str

class compilationSuccess(BaseModel):
    # The compiled output that concatenates and organizes both inputs without summarization
    compiled: Annotated[str, MinLen(1)] = Field(..., description='Compilation (not summary) of the provided agent outputs')
    references: List[str] = Field(..., description='List of all reference URLs from web agent')

class invalidRequest(BaseModel):
    error_message: str

summaryResponse: TypeAlias = Union[compilationSuccess, invalidRequest]

agent = Agent(
    model=OPENAI_MODEL,
    output_type=summaryResponse,
    output_retries=3,
    model_settings={'temperature': 0.1}
)

@agent.system_prompt
def system_prompt(ctx: RunContext[Deps]) -> str:
    return f"""
You are a compilation agent. Compile these inputs verbatim into the `compiled` field and return the web links as `references`.

Inputs (ctx.deps):
- action_agent_response_recommendationAnswer -> {ctx.deps.action_agent_response_recommendationAnswer}
- action_agent_response_docsAnswer       -> {ctx.deps.action_agent_response_docsAnswer}
- web_agent_response                      -> {ctx.deps.web_agent_response}

Rules:
1. Do NOT summarize or alter content. Preserve whitespace, punctuation, and line breaks; include values verbatim.
2. The `references` output must be the list from `web_agent_references` (use [] if empty). Keep `compiled` non-empty.

Be concise and exact.
"""

# def main():
#     deps = Deps(
#         action_agent_response_recommendationAnswer="To communicate effectively and create readiness, focus on quality over volume. Leaders should use closed-loop routines and surface unique information first, as quality has a stronger link to performance than raw frequency. Implement two-way communication cadences such as monthly AMAs and weekly '3x3' sessions (3 wins, 3 risks), and ensure questions are tracked and responded to publicly.",
#         action_agent_response_docsAnswer="Using orthodox methods of communication can increase the lack of receptiveness in a group. Instead, communications should be personalized, delivered by trusted managers, one-on-one, and empathic. Avoid official and formal methods to enhance receptiveness and readiness.",
#         web_agent_response="Creating readiness through communication involves fostering an environment where individuals or groups are prepared and willing to engage in change or new initiatives. Effective communication is key to building this readiness, as it helps clarify objectives, address concerns, and motivate stakeholders.\n\nTo communicate effectively for readiness, it is important to assess the current level of readiness among the audience. This involves understanding their perceptions, concerns, and motivations. Clear and transparent communication helps in aligning goals and expectations, reducing resistance, and encouraging adoption of new behaviors or changes. Additionally, creating a supportive environment where open dialogue is encouraged can help address fears and build trust, which are crucial for readiness.\n\nKey points:\n- Assess current readiness levels and perceptions.\n- Use clear and transparent communication to align goals.\n- Encourage open dialogue to address concerns and build trust.\n- Motivate stakeholders by highlighting benefits and addressing fears.",
#     )
#     response = agent.run_sync(deps=deps)

#     print(response.output)

# if __name__ == "__main__":
#     main()