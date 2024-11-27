from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter


class AILegislationAnalyzer:
    VIEWPOINT_DEFINITION = """
    A viewpoint refers to any distinct observation, argument, or perspective expressed in the documents about a specific subject. 
    Viewpoints must provide justification, evidence, or reasoning and are classified as Pro, Con, or Neutral based on their nature.
    """

    PRO_DEFINITION = """
    **Pro**: A Pro viewpoint highlights positive implications, benefits, or advantages of the subject. These include arguments showing how the subject solves problems, adds value, or emphasizes constructive outcomes.
    Examples:
    - Generative AI can increase productivity by automating repetitive tasks.
    - LLMs could enable greater decentralization by lowering communication barriers.
    """

    CON_DEFINITION = """
    **Con**: A Con viewpoint highlights negative implications, risks, or disadvantages of the subject. These include arguments that emphasize harm, inefficiencies, ethical dilemmas, or other adverse effects.
    Examples:
    - Generative AI could exacerbate misinformation by generating plausible fake content.
    - The high resource demands of training LLMs may contribute to environmental harm.
    """

    NEUTRAL_DEFINITION = """
    **Neutral**: A Neutral viewpoint is non-committal or fact-based without expressing a clear stance for or against the subject. These include statements describing trade-offs, complexities, or processes without bias.
    Examples:
    - Generative AI requires robust regulation to ensure ethical use.
    - The use of LLMs in organizations introduces both opportunities and challenges, depending on their application.
    """

    RESPONSE_TEMPLATE = f"""
    For all documents listed, identify all relevant pros, cons, and neutral viewpoints based on the following definitions:

    {VIEWPOINT_DEFINITION}

    {PRO_DEFINITION}

    {CON_DEFINITION}

    {NEUTRAL_DEFINITION}

    Use the following response template to extract points for each document:

    - **Document Identifier:** 
    - Pros:
        - ...
    - Cons:
        - ...
    - Neutral:
        - ...

    List every distinct point extracted from the documents, ensuring each is classified according to the definitions.
    """

    TRAINING_DATA_GUIDANCE = """
    Refer to the following training data for inspiration and alignment when extracting points from new documents. Ensure that new points reflect similar clarity and relevance to the topic.

    Training Data:

    - Document 1:
      - **Pros:**
        - Generative AI tools like DALL-E and ChatGPT may decentralize non-computation industries by empowering individuals relative to institutions.
        - Generative AI could enable the use of personal devices for computation, reducing reliance on centralized services.
        - LLMs could serve as 'universal APIs,' allowing on-the-fly API creation and interaction between different user interfaces without formal APIs.
        - Automated cooperative interoperability could become possible, allowing software to communicate through natural language and evolve specific interfaces dynamically.
        - LLMs could make adversarial interoperability easier and reduce the need for government interoperability mandates.
      - **Cons:**
        - Initially, LLMs and generative AI were thought to centralize computation, requiring significant capital investment that only large companies could afford.
      - **Neutral:**
        - LLMs are described as potentially centralizing computation while decentralizing other aspects of interaction and creation.
        - The centralization of LLMs might enable greater modularity and flexibility in platform interactions.
    """

    def __init__(self, urls, api_key):
        self.urls = urls
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.loader = WebBaseLoader(urls)
        self.llm = ChatOpenAI(model="gpt-4o", openai_api_key=api_key)
        self.simplellm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key)
        self.output_parser = StrOutputParser()
        self.text_splitter = RecursiveCharacterTextSplitter()
        self.vector_store = None

        self.prompt_template = ChatPromptTemplate.from_template(f"""
        You are an AI legislation policy-maker analyzing multiple documents. For all documents, extract every pro, con, and neutral viewpoint on generative AI. 
        Use the provided training data as guidance to ensure that your points reflect similar clarity, relevance, and quality. 
        Strictly use information found in the documents for all points.

        {AILegislationAnalyzer.RESPONSE_TEMPLATE}

        {AILegislationAnalyzer.TRAINING_DATA_GUIDANCE}

        <context>
        {{context}}
        </context>

        Question: {{input}}
        """)

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

    def analyze_document_individually(self, doc):
        print("Analyzing", doc.metadata["Document Identifier"])
        prompt = self.prompt_template.format(input="Extract pros, cons, and neutral viewpoints.", context=doc.page_content)
        c = self.llm | self.output_parser
        response = c.invoke(prompt)

        parsed_points = self.parse_points(response)
        return parsed_points

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

    def verify_point(self, point, search_results):
        print("Verifying", point)
        if not search_results:
            return point, "no", None  

        verification_prompt = f"Given the following search results:\n\n{search_results}\n\nIs the following point verified? {point}\n\nPlease respond with just 'yes' or 'no'."
        verify_chain = self.simplellm | self.output_parser
        response = verify_chain.invoke(verification_prompt)
        print(response)

        return point, response.strip().lower(), search_results 

    def verify_points_in_document(self, doc, extracted_points):
        print("Verifying", doc.metadata["Document Identifier"])
        verified_points = {
            'Pros': [],
            'Cons': [],
            'Neutral': []
        }

        for category, points in extracted_points.items():
            for point in points:
                search_results = self.vector_store.similarity_search(point)
                verified_point, verification_response, results = self.verify_point(point, search_results)
                
                if verification_response == "yes.":

                    doc_id = doc.metadata['Document Identifier']
                    verified_points[category].append((verified_point, doc_id))

        return verified_points

    def combine_verified_points(self, verified_points_list):
        print('Combining verified points...')
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

        response = "Combined verified points:\n"
        for category, points in combined_points.items():
            response += f"\n**{category.capitalize()}:**\n"
            for point, doc_ids in points.items():
                doc_ids_list = ', '.join(sorted(doc_ids))  
                response += f"- Documents {doc_ids_list}: {point}\n"

        return response

    def run_analysis(self):
        docs = self.load_documents()
        documents = self.split_documents(docs)

        self.vector_store = self.create_vector_store(documents)  
        retriever = self.get_retriever(self.vector_store)

        extracted_points_list = []

        for doc in docs:
            extracted_points = self.analyze_document_individually(doc)
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

    api_key = "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"
    analyzer = AILegislationAnalyzer(urls, api_key)
    final_result = analyzer.run_analysis()
    print(final_result)