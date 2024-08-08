import os
import sys
from typing import TypedDict, Literal
from dotenv import load_dotenv
from agents.helper import read_and_strip_file
from langgraph.graph import StateGraph, END
from agents import generate_article, Rephrase_article , passer , validator


class AgentState(TypedDict):
    agent_outcome: str

def save_result_to_file(result, filename):
    with open(filename, 'w') as f:
        f.write(str(result))

def main():
    load_dotenv()
    agent_prompt_path = os.path.join('meta_data', 'initial_input.txt')

    initial_input = read_and_strip_file(agent_prompt_path)

    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("evaluator", passer)
    workflow.add_node("Generate", generate_article)
    workflow.add_node("Rephrase", Rephrase_article)

    # Add Flow
    workflow.set_entry_point("evaluator")
    workflow.add_conditional_edges(
        "evaluator", validator, {"Generate": "Generate", "not_relevant": END}
    )
    workflow.add_edge("Generate", "Rephrase")
    workflow.add_edge("Rephrase", END)

    app = workflow.compile()
    initial_state = {"agent_outcome": initial_input}
    result = app.invoke(initial_state)  

    print("-"*100)
    print("Result:", result)

    # Save Result to output.txt
    save_result_to_file(result["agent_outcome"], "output.txt")

if __name__ == "__main__":
    main()
