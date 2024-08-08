from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
from ..helper import create_dict
import os
class AgentState(TypedDict):
    agent_outcome: str

class checker(BaseModel):
    """Binary score to determine if it's okay to move further with the topic at hand."""
    binary_score: str = Field(
        description="The piece of text that will be generated will either answer 'yes' or 'no.'"
    )

class Configuration:
    def __init__(self):
        self.config_path = os.path.join('meta_data', 'validator', 'config.json')
        self.system_path = os.path.join('meta_data', 'validator', 'system.txt')
        self.human_path = os.path.join('meta_data', 'validator', 'human.txt')
        self.config = create_dict(self.config_path, self.system_path, self.human_path)


class Evaluator:
    def __init__(self, config: Configuration):
        self.llm = ChatOpenAI(model=config.config['model'], temperature=config.config['temperature'], 
                                          max_tokens=config.config['max_tokens'])
        self.structured_llm = self.llm.with_structured_output(checker)
        self.system = config.config['system']
        self.human = config.config['human'] + "\n\n" + "{data}"
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [("system", self.system), ("human", self.human)]
        )
        self.evaluator = self.grade_prompt | self.structured_llm

    def invoke(self, data):
        return self.evaluator.invoke({"data": data})

def passer(state: AgentState) -> AgentState:
    return state

def validator(state: AgentState) -> Literal["Generate", "not_relevant"]:
    data = state["agent_outcome"]
    config = Configuration()
    evaluator = Evaluator(config)
    result = evaluator.invoke(data)
    print(f"Validation state: {state}")
    print(result)
    return "Generate" if result.binary_score == "yes" else "not_relevant"
