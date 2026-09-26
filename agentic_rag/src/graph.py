from langgraph.graph import StateGraph, START, END
from database import GraphState
import os
from dotenv import load_dotenv
from nodes import retrieve, grade_documents, web_search, generate, decide_to_generate

load_dotenv()

def create_agentic_rag_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("web_search", web_search)
    workflow.add_node("generate", generate)
    
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

if __name__ == "__main__":
    app = create_agentic_rag_graph()
    
    inputs = {"query": "What are the core updates regarding all Anthropic models performance metrics?"}
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished executing node: {key}")
            
    print("Final Result:")
    print(output["generate"]["generation"])
