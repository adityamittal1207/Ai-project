import json
import ast
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import agents
import prompts
from tools import parse_viewpoints, verify_points_in_document
from state import AgentGraphState, state


def create_graph(model=None, model_size = "large", openai_key=None, temperature=0, model_endpoint=None, stop=None, output_parser = None, doc = None, embeddings = None):
    graph = StateGraph(AgentGraphState)


    graph.add_node(
        "extractor", 
        lambda state: agents.ViewpointExtactorAgent(
            state=state,
            model=model,
            model_size = model_size,
            openai_key = openai_key,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
            output_parser=output_parser,
        ).invoke(
            prompt=prompts.VIEWPOINT_EXTRACTION_TEMPLATE, doc = state['doc']
        )
    )

    graph.add_node(
        "classifier", 
        lambda state: agents.ClassifierAgent(
            state=state,
            model=model,
            model_size = "small",
            openai_key = openai_key,
            stop=stop,
            model_endpoint=model_endpoint,
            temperature=temperature,
            output_parser=output_parser,
        ).invoke(
            prompt=prompts.CLASSIFICATION_TEMPLATE, viewpoints = state['viewpoints']
        )
    )

    graph.add_node(
        "viewpoint_parser_tool",
        lambda state: parse_viewpoints(
            state=state
        )
    )

    graph.add_node(
        "viewpoint_verifier",
        lambda state: verify_points_in_document(state=state, embeddings=embeddings, text_splitter=SemanticChunker(embeddings))
    )

    graph.add_node("end", lambda state: agents.EndNodeAgent(state).invoke())

    # Add edges to the graph
    graph.set_entry_point("extractor")
    graph.set_finish_point("end")
    graph.add_edge("extractor", "viewpoint_parser_tool")
    graph.add_edge("viewpoint_parser_tool", "classifier")
    graph.add_edge("classifier", "viewpoint_verifier")
    graph.add_edge("viewpoint_verifier", "end")

    return graph

def compile_workflow(graph):
    workflow = graph.compile()
    return workflow