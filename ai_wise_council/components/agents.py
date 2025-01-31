from typing import Any, Callable, ClassVar, Literal
from uuid import uuid4
from loguru import logger

from langgraph.types import Command
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from ai_wise_council.components.model_clients import get_model_client
from ai_wise_council.prompts.constructors import make_debate_system_prompt, make_judge_system_prompt
from ai_wise_council.prompts.instructions import (
    bad_faith_instructions,
    good_faith_instructions,
    judge_instructions,
)


class Agent(BaseModel):
    id_agent: int | None = None
    model: str
    temperature: float
    role: Literal["debater", "judge"]
    max_completion_tokens: int | None = 100

    _id_counter: ClassVar[int] = 111

    def __init__(self, **data):
        Agent._id_counter += 111
        data['id_agent'] = Agent._id_counter
        super().__init__(**data)


class JudgeAgent(Agent):
    specific_instructions: str | None = None

    def invoke(self, *args, **kwargs) -> str:
        logger.info(f"Judge agent {self.id_agent} invoked with model {self.model}")
        model_client = get_model_client(self.model, self.max_completion_tokens, self.temperature)
        react_agent = create_react_agent(
            model_client,
            tools=[],
            state_modifier=make_judge_system_prompt(
                instructions=judge_instructions
            ),
        )
        return react_agent.invoke(*args, **kwargs)
    
    def _process_debate(self, question: str, messages: list[BaseMessage]) -> str:
        logger.debug(f"Processing debate for judge {self.id_agent} with {len([message for message in messages if not isinstance(message, HumanMessage)])} messages")
        initial_message = f"Who is the truthful debater? <DEBATE QUESTION> {question} </DEBATE QUESTION> \nHISTORIC MESSAGES:\n"
        parsed_debate = [f"<DEBATER ID: {message.additional_kwargs['debater_id']}> {message.content} </DEBATER ID: {message.additional_kwargs['debater_id']}>\n" for message in messages if not isinstance(message, HumanMessage)]
        return initial_message + "".join(parsed_debate)

    def agentic_node(self, state: Any, **kwargs) -> Callable:
        logger.info(f"Judge {self.id_agent} evaluating debate")
        debate_prompt = self._process_debate(state["question"], state["messages"])
        result: dict[str, list[BaseMessage] | Any] = self.invoke({"messages": [HumanMessage(content=debate_prompt)]})
        return Command(
            update={
                "messages": result["messages"]
            },
            goto=END
        )

class DebaterAgent(Agent):
    subrole: Literal["good_faith", "bad_faith"] | None = None
    specific_instructions: str | None = None

    def invoke(self, *args, **kwargs) -> str:
        logger.info(f"Debater {self.id_agent} ({self.subrole}) invoked with model {self.model}")
        model_client = get_model_client(self.model, self.max_completion_tokens, self.temperature)
        react_agent = create_react_agent(
            model_client,
            tools=[],
            state_modifier=make_debate_system_prompt(
                role=self.subrole, instructions=self.specific_instructions
            ),
        )
        return react_agent.invoke(*args, **kwargs)
    
    def _process_first_message(self, initial_message: list[BaseMessage]) -> str:
        return f"STORY AND QUESTION: \n{initial_message[0].content}"
    
    def _process_conversation_history(self, conversation_history: list[BaseMessage]) -> str:
        parsed_history = []
        for message in conversation_history[1:]:
            if isinstance(message, HumanMessage):
                continue
            
            if message.additional_kwargs['debater_id'] == self.id_agent:
                parsed_history.append(
                    f"""
                    <YOUR PREVIOUS MESSAGE> {message.content} </YOUR PREVIOUS MESSAGE>\n
                    """
                )
            else:
                parsed_history.append(f"<ADVERSARY PREVIOUS MESSAGE> {message.content} </ADVERSARY PREVIOUS MESSAGE>\n")
            
        return "".join(parsed_history)
    
    def agentic_node(self, state: Any, **kwargs) -> Callable:
        logger.info(f"Debater {self.id_agent} ({self.subrole}) generating response. Remaining interventions: {state['total_interventions']}")
        # parse first message
        prompt = self._process_first_message(state["messages"])

        # parse conversation history
        if len(state["messages"]) > 1:
            prompt += self._process_conversation_history(state["messages"])

        # call the llm and tag the response with debater id metadata
        result: dict[str, list[BaseMessage] | Any] = self.invoke({"messages": [HumanMessage(content=prompt)]})
        result["messages"][-1].additional_kwargs["debater_id"] = self.id_agent
    
        return Command(
            update={
                "total_interventions": state["total_interventions"] - 1,
                "messages": result["messages"]
            },
        )


def create_debater(
    model: str,
    temperature: float,
    subrole: Literal["good_faith", "bad_faith"],
    id_agent: int | None = None,
) -> DebaterAgent:
    # load specific instructions based on subrole
    if subrole == "good_faith":
        specific_instructions = good_faith_instructions
    else:
        specific_instructions = bad_faith_instructions

    return DebaterAgent(
        model=model,
        temperature=temperature,
        role="debater",
        subrole=subrole,
        specific_instructions=specific_instructions,
        id_agent=id_agent,
    )


def create_judge(model: str, temperature: float) -> JudgeAgent:
    return JudgeAgent(model=model, temperature=temperature, role="judge")

