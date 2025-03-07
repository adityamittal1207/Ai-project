from langchain_openai import ChatOpenAI
import os

def get_large_open_ai(temperature=0, model='gpt-4o', openai_key = None):
    llm = ChatOpenAI(
    model=model,
    temperature = temperature,
    openai_api_key = openai_key,
)
    return llm
    
def get_small_open_ai(temperature=0, model='gpt-4o-mini', openai_key = None):

    llm = ChatOpenAI(
    model=model,
    temperature = temperature,
    openai_api_key = openai_key,
)
    return llm