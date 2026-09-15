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

class MessageClassifier(BaseModel):
    message_type: Literal["emotional", "logical"] = Field(
        ...,
        description="Classify if the message requires an emotional or logical type"
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_types: str | None


graph_builder = StateGraph(State)

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_llm = llm.with_structured_output(MessageClassifier)

    result = classifier_llm.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'emotional': if it asks for emotional support, therapy, deals with feelings
            - 'logical': if it asks for facts, information, logic
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type", "logical")
    if message_type == "emotional":
        return {"next": "therapist"}
    return {"next": "logical"}

def therapis_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": """You are a compassionate therapist. """
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}

def logical_agent(state: State):
    last_message = state["messages"][-1]
    messages = [
        {
            "role": "system",
            "content": """You are a purely logical assistant"""
        },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    reply = llm.invoke(messages)
    return {"messages": [{"role": "assistant", "content": reply.content}]}


graph_builder = StateGraph(State)

graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("therapist", therapist_agent)
graph_builder.add_node("logical", logical_agent)

graph_builder.add_edge(START, end_key:"classifier")
graph_builder.add_edge(start_key:"classifier", end_key:"router")
graph_builder.add_conditional_edges(
    source: "router",
    lambda state: state.get("next"),
    path_map: {"therapist": "therapist", "logical": "logical"}
)
graph_builder.add_edge(start_key:"therapist" ,END)
graph_builder.add_edge(start_key:"logical", END)

graph = graph_builder.compile()
