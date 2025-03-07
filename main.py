from graph import create_graph, compile_workflow
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings
import os



# server = 'ollama'
# model = 'llama3:instruct'
# model_endpoint = None

server = 'openai'
model = 'gpt-4o'
model_endpoint = None

iterations = 40


urls = [
    "https://outofcontrol.substack.com/p/is-ai-a-tradition-machine",
    "https://outofcontrol.substack.com/p/large-language-models-could-re-decentralize"
    ]

docs = WebBaseLoader(urls).load()

for i, doc in enumerate(docs):
    doc.metadata = {'Document Identifier': f"Document {i + 1}"}

print ("Creating graph and compiling workflow...")
openai_key = ""
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
graph = create_graph(model=model, model_endpoint=model_endpoint, output_parser=StrOutputParser(), openai_key=openai_key, embeddings=embeddings)
workflow = compile_workflow(graph)
print ("Graph and workflow created.")


if __name__ == "__main__":

    while True:
        
        # thread = {"configurable": {"thread_id": "4"}}

        # for event in workflow.stream(
        #     dict_inputs, thread, limit, stream_mode="values"
        #     ):
        #     if verbose:
        #         print("\nState Dictionary:", event)
        #     else:
        #         print("\n")

        for event in workflow.stream({"doc": docs[0]}):
            print("\n")



    