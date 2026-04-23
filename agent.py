import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph
from langgraph.constants import END
from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Define the state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    next: str

# Initialize MongoDB client for DocumentDB
client = MongoClient(os.getenv("DOCUMENTDB_URI"))
db = client["movie_db"]
collection = db["movies"]

# Initialize checkpointer
checkpointer = MongoDBSaver(client)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)

# Define tools
@tool
def description_search(query: str) -> str:
    """Search for books by description."""
    # Use CosmosDB vector search with cosmosSearch
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    results = list(collection.aggregate([
        {
            "$search": {
                "cosmosSearch": {
                    "vector": query_embedding,
                    "path": "embedding",
                    "k": 5,
                    "similarity": "COS"
                }
            }
        }
    ]))
    # Format results
    docs = [res.get("text", "No text") for res in results]
    return "\n".join(docs)

@tool
def title_search(query: str) -> str:
    """Search for books by title."""
    # Use MongoDB search index
    results = collection.find({"$text": {"$search": query}}, {"title": 1, "description": 1})
    return "\n".join([f"{r['title']}: {r['description']}" for r in results])

@tool
def save_memory(memory: str) -> str:
    """Save user preferences to memory."""
    # For simplicity, save to a memory collection
    memory_collection = db["memory"]
    memory_collection.insert_one({"session_id": "current", "memory": memory})
    return "Memory saved."

@tool
def retrieve_memories() -> str:
    """Retrieve user memories."""
    memory_collection = db["memory"]
    memories = memory_collection.find({"session_id": "current"})
    return "\n".join([m["memory"] for m in memories])

tools = [description_search, title_search, save_memory, retrieve_memories]

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)

# Define nodes
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response], "next": "tools" if response.tool_calls else END}

def tools_node(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = last_message.tool_calls
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        if tool_name == "description_search":
            result = description_search.invoke(tool_args)
        elif tool_name == "title_search":
            result = title_search.invoke(tool_args)
        elif tool_name == "save_memory":
            result = save_memory.invoke(tool_args)
        elif tool_name == "retrieve_memories":
            result = retrieve_memories.invoke({})
        results.append(result)
    return {"messages": messages + [AIMessage(content="\n".join(results))], "next": END}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_edge("agent", "tools")
graph.add_edge("tools", END)
graph.set_entry_point("agent")

# Compile with checkpointer
app = graph.compile(checkpointer=checkpointer)