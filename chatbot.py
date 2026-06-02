from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in answering questions about a pizza restaurant

Here are some revelevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("=============================")
    question = input('Ask your question (q to quit): ')
    print('\n')
    if question == 'q':
        break
    reviews = retriever.invoke(question) # retrieve relevant documents to our question
    # result = chain.invoke({"reviews": [], "question": question})
    result = chain.invoke({"reviews": reviews, "question": question})
    print(result)