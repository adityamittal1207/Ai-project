import json
import ast
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import agents as agents
import prompts
from tools import parse_viewpoints, verify_points_in_document
from state import AgentGraphState, state


def create_graph(api_keys = None, model = "", temperature=0, model_endpoint=None, stop=None, output_parser = None, doc = None, embeddings = None):
    graph = StateGraph(AgentGraphState)

    graph.add_node(
        "extractor", 
        lambda state: agents.ViewpointExtactorAgent(
            state=state,
            api_keys=api_keys,
            model = model,
            model_size="large",
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
            api_keys=api_keys,
            model = model,
            model_size="small",
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

    graph.add_node("end", lambda state: agents.EndNodeAgent(state=state, model=model).invoke())

    # Add edges to the graph
    graph.set_entry_point("extractor")
    graph.set_finish_point("end")
    graph.add_edge("extractor", "viewpoint_parser_tool")
    # graph.add_edge("viewpoint_parser_tool", "classifier")
    # graph.add_edge("classifier", "viewpoint_verifier")
    # graph.add_edge("viewpoint_verifier", "end")
    graph.add_edge("viewpoint_parser_tool", "end")

    return graph

def compile_workflow(graph):
    workflow = graph.compile()
    return workflow