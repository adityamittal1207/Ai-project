from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the list of URLs
urls = [
    "https://outofcontrol.substack.com/p/is-ai-a-tradition-machine",
    "https://outofcontrol.substack.com/p/large-language-models-could-re-decentralize",
    "https://outofcontrol.substack.com/p/the-intellect-of-things",
    "https://outofcontrol.substack.com/p/a-modest-proposal-for-regulating",
    'https://www.foxnews.com/opinion/forget-criticisms-ai-could-help-keep-children-safe-online',
    "https://www.foxnews.com/opinion/christians-shouldnt-fear-ai-should-partner-with-it"
]

output_parser = StrOutputParser()


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

List out every distinct point that you can extract from each document.
"""

# Modify the prompt to include the document source and ensure thorough coverage
prompt_template = ChatPromptTemplate.from_template(f"""
You are an AI legislation policy-maker analyzing multiple documents. For all 3 documents, extract every pro, con, and viewpoint on generative AI. Strictly use information found in the documents for all points. Follow the response template:

{response_template}

<context>
{{context}}
</context>

Question: {{input}}""")

# Function to analyze each document individually
def analyze_document_individually(llm, doc, prompt_template):
    prompt = prompt_template.format(input="Extract pros, cons, and viewpoints.", context=doc.page_content)
    response = llm.invoke(prompt)
    return response

# Function to verify that points are actually in the document
def verify_points_in_document(llm, doc, extracted_points):
    verification_prompt = f"Verify if the following points are present in this document:\n\n{extracted_points}\n\nDocument Content:\n{doc.page_content}"
    response = llm.invoke(verification_prompt)
    return response

# Function to combine the verified points from all documents
def combine_verified_points(llm, verified_points_list):
    combined_prompt = "Combine the following verified points from different documents into a single coherent analysis, ensuring each distinct point is listed and formatted clearly. For each distinct point, mention if this is viewed positively, negatively, or neutrally, and mention the document or documents used to back up the point. Elaborate each point and include every possible distinct point, include a very large number of points:\n\n"
    for idx, points in enumerate(verified_points_list):
        combined_prompt += f"Document {idx+1}:\n{points}\n\n"
    chain = llm | output_parser
    response = chain.invoke(combined_prompt)
    return response

# Analyze each document and verify the points
extracted_points_list = []
for doc in docs:
    extracted_points = analyze_document_individually(llm, doc, prompt_template)
    verified_points = verify_points_in_document(llm, doc, extracted_points)
    extracted_points_list.append(verified_points)

# Combine the verified points from all documents
final_result = combine_verified_points(llm, extracted_points_list)

# Print the final combined result
print(final_result)
