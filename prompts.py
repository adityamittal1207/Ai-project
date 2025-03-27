from langchain_core.prompts import ChatPromptTemplate

VIEWPOINT_EXTRACTION_BASE_TEMPLATE = """
    Extract all distinct viewpoints related to generative AI from the provided document. Viewpoints are observations, arguments, or perspectives relevant to the subject, and they should be expressed clearly with evidence or reasoning if present.
    Ensure the extracted viewpoints are fact-based and concise. Generate as many viewpoints as possible, and ensure to have enough viewpoints to minimize any ambiguity, with each viewpoint covering its own distinct topic.

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

    YOU MUST USE the following response format:
    - **Document Identifier:** 
    - Viewpoints:
        - ...
    """

VIEWPOINT_EXTRACTION_TEMPLATE = ChatPromptTemplate.from_template(f"""
    You are an AI legislation policy-maker analyzing multiple documents. Extract all viewpoints related to generative AI. Ensure that viewpoints are concise and fact-based.

    {VIEWPOINT_EXTRACTION_BASE_TEMPLATE}

    <context>
    {{context}}
    </context>

    Question: {{input}}
    """)

CLASSIFICATION_BASE_TEMPLATE = """
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

CLASSIFICATION_TEMPLATE = ChatPromptTemplate.from_template(f"""
    {CLASSIFICATION_BASE_TEMPLATE}

    Viewpoint: {{viewpoint}}
    """)