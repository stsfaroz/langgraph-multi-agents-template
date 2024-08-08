import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from ..helper import create_dict

class AgentState(TypedDict):
    agent_outcome: str

class Configuration:
    def __init__(self):
        self.config_path = os.path.join('meta_data', 'generate', 'config.json')
        self.system_path = os.path.join('meta_data', 'generate', 'system.txt')
        self.human_path = os.path.join('meta_data', 'generate', 'human.txt')
        self.config = create_dict(self.config_path, self.system_path, self.human_path)

class Rephraser:
    def __init__(self, config: Configuration):
        self.llm_to_rephrase = ChatOpenAI(model=config.config['model'], temperature=config.config['temperature'], 
                                          max_tokens=config.config['max_tokens'])
        self.Rephrasing_system = config.config['system']
        self.human = config.config['human'] + "\n\n" + "{data}"
        self.Rephrasing_prompt = ChatPromptTemplate.from_messages([("system", self.Rephrasing_system), ("human", self.human)])
        self.Rephrase = self.Rephrasing_prompt | self.llm_to_rephrase

    def invoke(self, data):
        return self.Rephrase.invoke({"data": data})

def Rephrase_article(state: AgentState) -> AgentState:
    config = Configuration()
    rephraser = Rephraser(config)
    print(f"Rephrase_article: Current state: {state}")
    data = state["agent_outcome"]
    result = rephraser.invoke(data)
    state["agent_outcome"] = result.content
    return state
