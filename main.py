from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.vectorstores.base import VectorStoreRetriever

# Define the list of URLs
urls = [
    "https://outofcontrol.substack.com/p/is-ai-a-tradition-machine",
    "https://outofcontrol.substack.com/p/large-language-models-could-re-decentralize",
    "https://outofcontrol.substack.com/p/the-intellect-of-things",
]

# Load documents with metadata
embeddings = OpenAIEmbeddings(openai_api_key="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")

loader = WebBaseLoader(urls)
docs = loader.load()

# Add metadata to documents after loading
for i, doc in enumerate(docs):
    doc.metadata = {'Document Identifier': f"Document {i+1}"}

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

retriever = vector.as_retriever()

llm = ChatOpenAI(openai_api_key="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")

output_parser = StrOutputParser()

# Define the response template to enforce the structure
response_template = """
For all 3 documents listed, identify all relevant pros, cons, and viewpoints. Ensure each document is addressed separately.

- **Document Identifier:** 
  - Pros:
    - ...
  - Cons:
    - ...
  - Viewpoints:
    - ...

List out every point that you can extract from each document.
"""

# Modify the prompt to include the document source and ensure thorough coverage
prompt = ChatPromptTemplate.from_template(f"""
You are an AI legislation policy-maker analyzing multiple documents. For all 3 documents, extract every pro, con, and viewpoint on generative AI. Strictly use information found in the documents for all points. Follow the response template:

{response_template}

<context>
{{context}}
</context>

Question: {{input}}""")

# Create a document chain that iterates over documents to ensure full extraction
document_chain = create_stuff_documents_chain(llm, prompt, output_parser=output_parser)

# Use the retrieval chain to combine results effectively
retrieval_chain = create_retrieval_chain(retriever, document_chain)

result = retrieval_chain.invoke({"input": "Determine the viewpoints, pros, and cons of the 3 documents. Refer to the documents by their document identifier in their metadata."})

# Parse and print the answer, which includes the document source
print(result['answer'])
