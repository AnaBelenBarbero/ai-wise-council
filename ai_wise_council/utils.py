from typing import Any, Callable, Sequence
import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import BaseChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

model_openai = ChatOpenAI(model="gpt-4o-mini", max_completion_tokens=100)
model_anthropic = ChatAnthropic(model="claude-3-haiku-20240307")
model_deepseek = BaseChatOpenAI(model="deepseek-chat",
                                openai_api_key=DEEPSEEK_API_KEY,
                                openai_api_base='https://api.deepseek.com',
                                max_tokens=100)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    prompt_template: ChatPromptTemplate
    model_used: BaseChatModel
    language: str


class SummarizerState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    prompt_template: ChatPromptTemplate
    model_used: BaseChatModel
    language: str
    output_experts: list[dict[str, list[BaseMessage]]]


def trimmer(model: BaseChatModel) -> Callable[[MessagesState], list[BaseMessage]]:
    return trim_messages(
        max_tokens=50,
        strategy="last",
        token_counter=model,
        include_system=True,
        allow_partial=True,
        start_on="human",
    )


def call_first_layer_model(state: State) -> dict[str, list[BaseMessage]]:
    trimmed_messages = trimmer(model=state["model_used"]).invoke(state["messages"])
    template = state["prompt_template"]
    model = state["model_used"]
    prompt = template.invoke(
        {"messages": trimmed_messages, "language": state.get("language", "English")},
        {}, # config
    )
    response = model.invoke(prompt)
    return {"messages": [response]}


def call_second_layer_model(state: SummarizerState) -> dict[str, list[BaseMessage]]:
    trimmed_messages = trimmer(model=state["model_used"]).invoke(state["messages"])
    template = state["prompt_template"]
    model = state["model_used"]
    output_experts = "".join(
        [
            f" Expert {i+1}: {expert['messages'][-1].content}"
            for i, expert in enumerate(state["output_experts"])
        ]
    )

    prompt = template.invoke(
        {
            "messages": trimmed_messages,
            "language": state.get("language", "English"),
            "output_experts": output_experts,
        },
        {}, # config
    )
    response = model.invoke(prompt)

    output_summarizer = f"""The experts have provided the following information: {output_experts}
        The summary of the information is: {response}
    """
    return {"messages": [output_summarizer]}


def create_graph(
    schema: dict[str, Any], function: Callable[[State], dict[str, list[BaseMessage]]]
):
    workflow = StateGraph(state_schema=schema)
    workflow.add_edge(START, "llm_call")
    workflow.add_node("llm_call", function)
    return workflow.compile(checkpointer=MemorySaver())
