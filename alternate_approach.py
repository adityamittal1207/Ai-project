from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

class AILegislationAnalyzer:
    VIEWPOINT_EXTRACTION_TEMPLATE = """
    Extract all distinct viewpoints related to generative AI from the provided document. Viewpoints are observations, arguments, or perspectives relevant to the subject, and they should be expressed clearly with evidence or reasoning if present.
    Ensure the extracted viewpoints are fact-based and concise.

    Example viewpoints:
    - Generative AI can streamline customer support by providing instant and accurate responses to common inquiries.
    - The use of generative AI raises privacy concerns due to its reliance on large datasets, which may include sensitive information.
    - Generative AI requires significant computational resources, leading to potential environmental impacts from energy consumption.
    - The development of generative AI could create new job opportunities in AI training and maintenance industries.
    - Concerns exist about the ethical implications of using generative AI in creative fields, such as art and writing, where it might replace human creators.
    - The adaptability of generative AI makes it a versatile tool for personalized learning experiences in education.
    - Regulations are needed to ensure that generative AI is used responsibly and does not perpetuate biases in its training data.
    - Generative AI could potentially disrupt traditional industries by enabling automation of complex tasks, such as medical diagnosis.
    - There are questions about the accountability of AI-generated content and the difficulty in attributing authorship.

    Use the following response format:
    - **Document Identifier:** 
    - Viewpoints:
        - ...
    """

    CLASSIFICATION_TEMPLATE = """
    **Pro**: Highlights positive implications, benefits, or advantages of the subject.
    **Con**: Highlights negative implications, risks, or disadvantages of the subject.
    **Neutral**: Non-committal or fact-based without a clear stance for or against the subject. 

    Examples:
    Pro: The development of AI governance frameworks can foster responsible innovation.  
    Con: The use of AI in defense raises questions about autonomous decision-making in warfare. 
    Neutral: Human judgment remains paramount in intelligence work despite AI assistance.  
    Con: Training LLMs demands high resources, potentially harming the environment.  
    Pro: Large language models (LLMs) may decentralize industries by lowering communication barriers.  
    Con: AI's environmental footprint is becoming a public concern.  
    Pro: Generative AI requires robust regulation to ensure ethical use.  
    Neutral: The originality of AI-generated art is debated in the context of intellectual property.  
    Con: Relying on AI summaries may lead to the loss of nuanced understanding in complex cases.  
    Pro: Collaborative efforts between AI and humans can drive sustainable innovation.  
    Pro: AI's limitations highlight the importance of human creativity and judgment.  
    Con: Expanding AI surveillance raises ethical issues about privacy and civil liberties.  
    
    Respond with only the classification: Pro, Con, or Neutral.
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

        self.extraction_prompt_template = ChatPromptTemplate.from_template(f"""
        You are an AI legislation policy-maker analyzing multiple documents. Extract all viewpoints related to generative AI. Ensure that viewpoints are concise and fact-based.

        {AILegislationAnalyzer.VIEWPOINT_EXTRACTION_TEMPLATE}

        <context>
        {{context}}
        </context>

        Question: {{input}}
        """)

        self.classification_prompt_template = ChatPromptTemplate.from_template(f"""
        {AILegislationAnalyzer.CLASSIFICATION_TEMPLATE}

        Viewpoint: {{viewpoint}}
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

    def extract_viewpoints(self, doc):
        print("Extracting viewpoints from", doc.metadata["Document Identifier"])
        prompt = self.extraction_prompt_template.format(input="Extract all viewpoints.", context=doc.page_content)
        c = self.llm | self.output_parser
        response = c.invoke(prompt)

        viewpoints = self.parse_viewpoints(response)
        return viewpoints

    def classify_viewpoints(self, viewpoints):
        classified_points = {
            'Pros': [],
            'Cons': [],
            'Neutral': []
        }

        for viewpoint in viewpoints:
            prompt = self.classification_prompt_template.format(viewpoint=viewpoint)
            # print(prompt)
            c = self.simplellm | self.output_parser
            response = c.invoke(prompt)
            classification = response.strip()

            # print(viewpoint, classification)

            if classification == "Pro":
                classified_points['Pros'].append(viewpoint)
            elif classification == "Con":
                classified_points['Cons'].append(viewpoint)
            elif classification == "Neutral":
                classified_points['Neutral'].append(viewpoint)
        
        print(classified_points)

        return classified_points

    def parse_viewpoints(self, raw_input):
        viewpoints = []
        lines = raw_input.strip().split('\n')

        for line in lines:
            line = line.strip().replace("**", "")

            if line.startswith('- Viewpoints:'):
                continue
            elif line.startswith('- '):
                viewpoint = line[2:].strip()
                viewpoints.append(viewpoint)

        print(viewpoints)

        return viewpoints

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
                        combined_points[category][point] = []
                    combined_points[category][point].append(doc_id)

        return combined_points

    def format_points(self, combined_points):
        formatted_output = []

        for category, points in combined_points.items():
            formatted_output.append(f"{category}:\n" + "-" * len(category))
            for point, documents in points.items():
                formatted_output.append(f"- {point} (Source: {', '.join(documents)})")
            formatted_output.append("\n")

        return "\n".join(formatted_output)

    def run_analysis(self):
        print("Loading documents...")
        documents = self.load_documents()

        print("Splitting documents...")
        split_docs = self.split_documents(documents)

        print("Creating vector store...")
        self.vector_store = self.create_vector_store(split_docs)

        verified_points_list = []

        for doc in documents:
            print(f"Processing {doc.metadata['Document Identifier']}...")

            viewpoints = self.extract_viewpoints(doc)
            classified_points = self.classify_viewpoints(viewpoints)
            verified_points = self.verify_points_in_document(doc, classified_points)

            verified_points_list.append(verified_points)

        combined_results = self.combine_verified_points(verified_points_list)
        return self.format_points(combined_results)

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
