from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AILegislationAnalyzer:
    def __init__(self, urls, api_key):
        self.urls = urls
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.loader = WebBaseLoader(urls)
        self.llm = ChatOpenAI(openai_api_key=api_key)
        self.output_parser = StrOutputParser()
        self.text_splitter = RecursiveCharacterTextSplitter()

    def load_documents(self):
        docs = self.loader.load()

        for i, doc in enumerate(docs):
            doc.metadata = {'Document Identifier': f"Document {i + 1}"}
        return docs

    def split_documents(self, docs):
        return self.text_splitter.split_documents(docs)

    def create_vector_store(self, documents):
        return FAISS.from_documents(documents, self.embeddings)

    def get_retriever(self, vector_store):
        return vector_store.as_retriever()

    def parse_points(self, raw_input):
        points = {
            'Pros': [],
            'Cons': [],
            'Neutral': []
        }

        lines = raw_input.strip().split('\n')

        current_category = None

        for line in lines:
            line = line.strip()

            if line.startswith('- Pros:'):
                current_category = 'Pros'
            elif line.startswith('- Cons:'):
                current_category = 'Cons'
            elif line.startswith('- Neutral:'):
                current_category = 'Neutral'
            elif line.startswith('- ') and current_category:
                point = line[2:].strip()  
                points[current_category].append(point)

        return points

    def analyze_document_individually(self, doc, prompt_template):
        print("Analyzing", doc.metadata["Document Identifier"])
        prompt = prompt_template.format(input="Extract pros, cons, and neutral viewpoints.", context=doc.page_content)
        c = self.llm | self.output_parser
        response = c.invoke(prompt)

        parsed_points = self.parse_points(response)
        return parsed_points

    def verify_point(self, point, search_results):
        print(" verifying", point)
        if not search_results:
            return point, "no", None  

        verification_prompt = f"Given the following search results:\n\n{search_results}\n\nIs the following point verified? {point}\n\nPlease respond with just 'yes' or 'no'."
        verify_chain = self.llm | self.output_parser
        response = verify_chain.invoke(verification_prompt)

        return point, response.strip().lower(), search_results 

    def verify_points_in_document(self, doc, extracted_points):
        print("verifying", doc.metadata["Document Identifier"])
        verified_points = {
            'Pros': [],
            'Cons': [],
            'Neutral': []
        }

        for category, points in extracted_points.items():
            for point in points:
                search_results = self.vector_store.similarity_search(point)
                verified_point, verification_response, results = self.verify_point(point, search_results)
                
                if verification_response == "yes":

                    doc_id = doc.metadata['Document Identifier']
                    verified_points[category].append((verified_point, doc_id))

        return verified_points

    def combine_verified_points(self, verified_points_list):
        print('Combining verified points...')
        combined_prompt = "Combine all the verified points from the documents. Ensure to list them under their respective categories: Pros, Cons, and Neutral. Identify all documents for each point. Only include distinct points. Merge similar points and cite both documents. Format the response as follows:\n\n"

        combined_prompt += """
        **Pros:**
        - Points from Document Identifier(s):
            - Merged / Standalone Point 1
            - Merged / Standalone Point 2
        **Cons:**
        - Points from Document Identifier(s):
            - Merged / Standalone Point 1
            - Merged / Standalone Point 2
        **Neutral:**
        - Points from Document Identifier(s):
            - Merged / Standalone Point 1
            - Merged / Standalone Point 2
        """

        combined_points = {
            'Pros': {},
            'Cons': {},
            'Neutral': {}
        }

        for idx, points in enumerate(verified_points_list):
            for category, verified_points in points.items():
                for point, doc_id in verified_points:

                    if point not in combined_points[category]:
                        combined_points[category][point] = {doc_id}
                    else:
                        combined_points[category][point].add(doc_id)

        for category, points in combined_points.items():
            combined_prompt += f"\n**{category.capitalize()}:**\n"
            for point, doc_ids in points.items():
                doc_ids_list = ', '.join(sorted(doc_ids))  
                combined_prompt += f"- Documents {doc_ids_list}: {point}\n"

        chain = self.llm | self.output_parser
        response = chain.invoke(combined_prompt)
        return response

    def run_analysis(self, prompt_template):

        docs = self.load_documents()
        documents = self.split_documents(docs)

        self.vector_store = self.create_vector_store(documents)  
        retriever = self.get_retriever(self.vector_store)

        extracted_points_list = []

        for doc in docs:
            extracted_points = self.analyze_document_individually(doc, prompt_template)
            verified_points = self.verify_points_in_document(doc, extracted_points)
            extracted_points_list.append(verified_points)

        final_result = self.combine_verified_points(extracted_points_list)
        return final_result


if __name__ == '__main__':

    urls = [
        "https://outofcontrol.substack.com/p/is-ai-a-tradition-machine",
        "https://outofcontrol.substack.com/p/large-language-models-could-re-decentralize",
        "https://outofcontrol.substack.com/p/the-intellect-of-things",
        "https://outofcontrol.substack.com/p/a-modest-proposal-for-regulating",
        'https://www.foxnews.com/opinion/forget-criticisms-ai-could-help-keep-children-safe-online',
        "https://www.foxnews.com/opinion/christians-shouldnt-fear-ai-should-partner-with-it"
    ]

    response_template = """
    For all 3 documents listed, identify all relevant pros, cons, and neutral viewpoints. Ensure each document is addressed separately.

    - **Document Identifier:** 
    - Pros:
        - ...
    - Cons:
        - ...
    - Neutral:
        - ...

    List out every distinct point that you can extract from each document.
    """

    prompt_template = ChatPromptTemplate.from_template(f"""
    You are an AI legislation policy-maker analyzing multiple documents. For all 3 documents, extract every pro, con, and neutral viewpoint on generative AI. Strictly use information found in the documents for all points. Follow the response template:

    {response_template}

    <context>
    {{context}}
    </context>

    Question: {{input}}""")

    api_key = "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"
    analyzer = AILegislationAnalyzer(urls, api_key)
    final_result = analyzer.run_analysis(prompt_template)

    print(final_result)

