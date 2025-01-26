from typing import Callable, Literal
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.state import CompiledStateGraph
from langchain_core.messages import HumanMessage

import random


# shared state
class State(MessagesState):
    total_interventions: int
    question: str


# conditional edges
def random_choice(state: State) -> Literal["good_faith_agent", "bad_faith_agent"]:
    return random.choice(["good_faith_agent", "bad_faith_agent"])


def should_continue(state: State) -> Literal["continue_debate", "judge"]:
    """Determine if the debate should continue or move to judgment"""
    if state["total_interventions"] > 0:
        return "continue_debate"
    return "judge_debate"


def create_debate(
    debater_good_faith: Callable, debater_bad_faith: Callable, judge: Callable
) -> CompiledStateGraph:
    builder = StateGraph(State)
    builder.add_node(
        "good_faith_agent",
        debater_good_faith,
    )
    builder.add_node(
        "bad_faith_agent",
        debater_bad_faith,
    )
    builder.add_node("judge_debate", judge)

    builder.add_conditional_edges(
        START,
        random_choice,
        {
            "good_faith_agent": "good_faith_agent",
            "bad_faith_agent": "bad_faith_agent",
        },
    )

    builder.add_conditional_edges(
        "good_faith_agent",
        should_continue,
        {
            "continue_debate": "bad_faith_agent",
            "judge_debate": "judge_debate",
        },
    )

    builder.add_conditional_edges(
        "bad_faith_agent",
        should_continue,
        {
            "continue_debate": "good_faith_agent",
            "judge_debate": "judge_debate",
        },
    )
    builder.add_edge("judge_debate", END)

    return builder.compile()


def run_debate(
    debate: CompiledStateGraph, story: str, question: str, total_interventions: int
):
    return debate.invoke(
        {
            "messages": [HumanMessage(content=f"STORY: {story}\nQUESTION: {question}")],
            "total_interventions": total_interventions,
            "question": question,
        }
    )
