import sys
from rag_engine import ScalableRAGEngine

def main():
    print("Initializing scalable local execution cluster services...")
    try:
        engine = ScalableRAGEngine(chroma_host="localhost", chroma_port=8000)
    except Exception as e:
        print(f"Initialization Failed. Verify that Chroma and Ollama instances are accessible: {e}")
        sys.exit(1)

    print("\n--- Local scalable RAG System Online ---")
    while True:
        user_query = input("\nEnter your question (or type 'exit' to quit): ")
        if user_query.strip().lower() == 'exit':
            break
            
        if not user_query.strip():
            continue

        print("Searching structural knowledge index and compiling context response...")
        try:
            response = engine.query(user_query)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"An execution runtime fault occurred processing query string: {e}")

if __name__ == "__main__":
    main()
