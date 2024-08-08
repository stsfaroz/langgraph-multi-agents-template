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

class Generator:
    def __init__(self, config: Configuration):
        self.llm_to_generate = ChatOpenAI(model=config.config['model'], temperature=config.config['temperature'], 
                                          max_tokens=config.config['max_tokens'])
        self.Generating_system = config.config['system']
        self.human = config.config['human'] + "\n\n" + "{data}"
        self.Generating_prompt = ChatPromptTemplate.from_messages([("system", self.Generating_system), ("human", self.human)])
        self.Generate = self.Generating_prompt | self.llm_to_generate

    def invoke(self, data):
        return self.Generate.invoke({"data": data})

def generate_article(state: AgentState) -> AgentState:
    config = Configuration()
    generator = Generator(config)
    print(f"Generating_article: Current state: {state}")
    data = state["agent_outcome"]
    result = generator.invoke(data)
    state["agent_outcome"] = result.content
    return state
