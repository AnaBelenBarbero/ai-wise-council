import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
AVAILABLE_MODELS: list[str] = ["gpt-4o-mini", "claude-3-haiku-20240307", "deepseek-chat"]


def get_model_client(model_name: str, max_completion_tokens: int = 700, temperature: float = 0.5) -> BaseChatOpenAI:
    if model_name == "gpt-4o-mini":
        return ChatOpenAI(model=model_name, max_completion_tokens=max_completion_tokens, temperature=temperature)
    elif model_name == "claude-3-haiku-20240307":
        return ChatAnthropic(model="claude-3-haiku-20240307", temperature=temperature)
    elif model_name == "deepseek-chat":
        return BaseChatOpenAI(
            model="deepseek-chat",
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base='https://api.deepseek.com',
            max_tokens=max_completion_tokens,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
