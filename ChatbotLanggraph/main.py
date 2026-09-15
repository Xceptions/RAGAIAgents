import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_ollama.llms import OllamaLLM


script_dir = Path(__file__).resolve().parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)
api_key = os.getenv("API_KEY")

llm_model = init_chat_model(
    "anthropic:claude-3-5-sonnet-latest",
    temperature = 0
)

class State(TypedDict):
    