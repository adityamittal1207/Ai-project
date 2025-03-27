import models
import prompts
from state import AgentGraphState
import csv

class Agent:
    def __init__(self, state: AgentGraphState, model = "", model_size = "", api_keys = None, temperature=0, model_endpoint=None, stop=None, output_parser = None, doc = None):
        self.state = state
        self.model = model
        self.model_size = model_size
        self.temperature = temperature
        self.model_endpoint = model_endpoint
        self.stop = stop
        self.api_keys = api_keys
        self.output_parser = output_parser

    def get_llm(self):
        return models.get_model(self.model, api_keys=self.api_keys, temperature=0.05) if self.model_size == "large" else models.get_model("gpt-4o-mini", api_keys=self.api_keys)

    def update_state(self, key, value):
        self.state = {**self.state, key: value}

class ViewpointExtactorAgent(Agent):
    def invoke(self, prompt=prompts.VIEWPOINT_EXTRACTION_TEMPLATE, doc = None):

        extractor_prompt = prompt.format(input="Extract all viewpoints.", context=doc.page_content)

        llm = self.get_llm()

        print(llm)

        # print(extractor_prompt)

        invoke_chain = llm | self.output_parser

        response = invoke_chain.invoke(extractor_prompt)

        self.update_state("viewpoint_extractor_response", response)
        print(f"Viewpoint Extractor: {response}")
        return self.state
    
class ClassifierAgent(Agent):
    def invoke(self, prompt = prompts.CLASSIFICATION_TEMPLATE, viewpoints = None):
        classified_points = {
            'Pros': [],
            'Cons': [],
            'Neutral': []
        }

        for viewpoint in viewpoints:
            classifier_prompt = prompt.format(viewpoint=viewpoint)
            llm = self.get_llm()
            c = llm | self.output_parser

            # print(classifier_prompt)
            response = c.invoke(classifier_prompt)
            classification = response.strip()

            # print(classification)

            if classification == "Pro":
                classified_points['Pros'].append(viewpoint)
            elif classification == "Con":
                classified_points['Cons'].append(viewpoint)
            elif classification == "Neutral":
                classified_points['Neutral'].append(viewpoint)
            
        self.update_state("classified_viewpoints", classified_points)

        print(f"Classifier: {classified_points}")
        
        return self.state

class EndNodeAgent(Agent):
    def invoke(self):
        filepath = "responses.csv"
        with open(filepath, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows([[self.model, self.state["viewpoints"]]])

        self.update_state("end_chain", "end_chain")
        return self.state