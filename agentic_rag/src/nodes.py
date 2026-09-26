import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from database import GraphState, get_vector_store

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieve(state: GraphState):
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    docs = retriever.invoke(state["query"])
    return {"documents": docs, "query": state["query"], "loop_count": state.get("loop_count", 0)}

def grade_documents(state: GraphState):
    docs = state["documents"]
    query = state["query"]
    
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a grader assessing relevance of a retrieved document to a user question.\n"
                   "Respond with a JSON object containing a single key 'score' which must be either 'yes' or 'no'."),
        ("human", "Document: {document}\nQuestion: {query}")
    ])
    
    structured_llm = llm.with_structured_output(dict)
    
    filtered_docs = []
    web_fallback = False
    
    for doc in docs:
        res = structured_llm.invoke(grader_prompt.format_messages(document=doc.page_content, query=query))
        if res.get("score") == "yes":
            filtered_docs.append(doc)
        else:
            web_fallback = True # Trigger fallback if any chunk is irrelevant
            
    return {"documents": filtered_docs, "web_fallback": web_fallback}

def web_search(state: GraphState):
    query = state["query"]
    docs = state["documents"]
    
    web_search_tool = TavilySearchResults(max_results=2)
    search_results = web_search_tool.invoke({"query": query})
    
    for res in search_results:
        docs.append(Document(page_content=res["content"], metadata={"source": res["url"]}))
        
    return {"documents": docs}

def generate(state: GraphState):
    query = state["query"]
    docs = state["documents"]
    
    context = "\n\n".join([d.page_content for d in docs])
    
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Answer the question using ONLY the provided context. Cite sources if available."),
        ("human", "Context:\n{context}\n\nQuestion: {query}")
    ])
    
    chain = rag_prompt | llm
    response = chain.invoke({"context": context, "query": query})
    
    return {"generation": response.content}

def decide_to_generate(state: GraphState):
    # Infinite loop protection
    if state.get("loop_count", 0) >= 2:
        print("-> Loop limit hit. Forcing generation.")
        return "generate"
        
    if state["web_fallback"]:
        print("-> Irrelevant data found. Routing to Web Search.")
        return "web_search"
    
    print("-> Data sufficient. Routing to Generation.")
    return "generate"
