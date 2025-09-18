from pymilvus import MilvusClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from typing_extensions import TypedDict, Annotated, Optional
from langchain_core.messages import HumanMessage

from agents.action_recommendation import agent as action_agent
from agents.action_recommendation import Deps as ActionDeps
from agents.action_recommendation import actionSuccess
from agents.web_resources import agent as web_agent
from agents.web_resources import Deps as WebDeps
from agents.web_resources import webSuccess
from agents.summary import agent as compilation_agent
from agents.summary import Deps as SummaryDeps
from agents.summary import compilationSuccess

class AllState(TypedDict):
    request: Annotated[list[AnyMessage], add_messages]

    action_agent_response_recommendationAnswer: Annotated[Optional[str], "The response from the action recommendation agent"]
    action_agent_response_docsAnswer: Annotated[Optional[str], "The response from the action recommendation agent"]
    web_agent_response: Annotated[Optional[str], "The response from the web resources agent"]
    web_agent_references: Annotated[Optional[list[str]], "The list of reference URLs from the web resources agent"]
    compilation_agent_response: Annotated[Optional[str], "The final compiled response to the user"]

    action_agent_error: Annotated[Optional[str], "Error message from the action recommendation agent if any"]
    web_agent_error: Annotated[Optional[str], "Error message from the web resources agent if any"]
    compilation_agent_error: Annotated[Optional[str], "Error message from the compilation agent if any"]

def human_entry_node(state: AllState):
    # Human entry node: single entry point for a human request.
    # This node ensures the state's `request` list contains the latest HumanMessage.
    # If the graph runtime provides an incoming HumanMessage as the last message in state,
    # keep it. Otherwise, ensure there's at least an empty list to satisfy the schema.
    # (The runtime should populate `state["request"]` from the START payload where possible.)
    request_list = state.get("request") or []
    # If runtime provided a raw human string under "human_input", normalize it into a HumanMessage
    if not request_list and state.get("human_input"):
        request_list = [HumanMessage(content=state["human_input"])]
    return {"request": request_list}

def action_recommendation_node(state: AllState):
    action_agent_response = action_agent.run_sync(
        user_prompt=state["request"][-1].content if state["request"] else "",
        deps=ActionDeps(client=MilvusClient(uri="./milvus_tgps.db"))
    )

    if isinstance(action_agent_response.output, actionSuccess):
        return {
            "action_agent_response_recommendationAnswer": action_agent_response.output.recommendationAnswer,
            "action_agent_response_docsAnswer": action_agent_response.output.docsAnswer,
        }
    else:
        return {
            "action_agent_error": action_agent_response.output.error_message
        }
    
def web_node(state: AllState):
    web_agent_response = web_agent.run_sync(
        user_prompt=state["request"][-1].content if state["request"] else "",
        deps=WebDeps(client=MilvusClient(uri="./milvus_tgps.db"))
    )

    if isinstance(web_agent_response.output, webSuccess):
        return {
            "web_agent_response": web_agent_response.output.webAnswer,
            "web_agent_references": web_agent_response.output.references,
        }
    else:
        return {
            "web_agent_error": web_agent_response.output.error_message
        }
    
def compilation_node(state: AllState):
    compilation_response = compilation_agent.run_sync(
        user_prompt="Compile the inputs.",
        deps=SummaryDeps(
            action_agent_response_recommendationAnswer=state.get("action_agent_response_recommendationAnswer", "") or "",
            action_agent_response_docsAnswer=state.get("action_agent_response_docsAnswer", "") or "",
            web_agent_response=state.get("web_agent_response", "") or ""
        )
    )

    if isinstance(compilation_response.output, compilationSuccess):
        return {
            "compilation_agent_response": compilation_response.output.compiled
        }
    else:
        return {
            "compilation_agent_error": compilation_response.output.error_message
        }
    
def create_graph():
    # instantiate the graph with the AllState schema
    # StateGraph expects the state schema as a constructor argument
    graph = StateGraph(state_schema=AllState)

    # register nodes
    graph.add_node("human_entry_node", human_entry_node)
    graph.add_node("action_recommendation_node", action_recommendation_node)
    graph.add_node("web_node", web_node)
    graph.add_node("compilation_node", compilation_node)

    # wiring:
    # START -> human_entry_node -> {action, web} -> compilation -> END
    graph.set_entry_point("human_entry_node")
    graph.add_edge("human_entry_node", "action_recommendation_node")
    graph.add_edge("human_entry_node", "web_node")

    graph.add_edge("action_recommendation_node", "compilation_node")
    graph.add_edge("web_node", "compilation_node")

    graph.add_edge("compilation_node", END)

    # return the graph instance; do not compile here so callers can control lifecycle
    return graph.compile()

graph = create_graph()

def create_graph_image():

    from langchain_core.runnables.graph import MermaidDrawMethod

    graph_png = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.PYPPETEER,)

    with open("graph.png", "wb") as f:
        f.write(graph_png)

def main():
    initial_state = {
                        "request": [
                            HumanMessage(content="How to communicate to create readiness?")
                        ],
                    }

    for event in graph.stream(initial_state):
        for key in event:
            print("\n-----------------------------------")
            print("Done with " + key)
            print("\n***********************************\n")

if __name__ == "__main__":
    # create_graph_image()
    main()