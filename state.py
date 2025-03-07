from typing import TypedDict, Annotated
from langchain_core.documents import Document
from langgraph.graph.message import add_messages

# Define the state object for the agent graph
class AgentGraphState(TypedDict):
    viewpoint_extractor_response: str
    viewpoints: list    
    end_chain: list
    classified_viewpoints: list
    verified_points: list
    doc: Document
    # planner_response: Annotated[list, add_messages]

state = {
    "viewpoint_extractor_response":"",
    "viewpoints": [],
    "end_chain": [],
    "classified_viewpoints": [],
    "verified_points": [],
    "doc": None,
}