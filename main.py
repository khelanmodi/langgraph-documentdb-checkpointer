import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ensure OPENAI_API_KEY is set in your environment or .env file

from agent import app, collection, db
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from langchain_core.messages import HumanMessage, AIMessage

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Load and process data
loader = JSONLoader(file_path="books.json", jq_schema=".[] | \"Title: \\(.title)\\nDescription: \\(.description)\"")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Insert documents with embeddings into DocumentDB
print("Inserting documents with embeddings...")
for doc in docs:
    embedding = embeddings.embed_query(doc.page_content)
    collection.insert_one({
        "text": doc.page_content,
        "embedding": embedding,
        "metadata": doc.metadata
    })
print(f"Inserted {len(docs)} documents.")

print("Creating vector search index...")
# Create vector search index
try:
    collection.create_index(
        [("embedding", "cosmosSearch")],
        name="vector_index",
        cosmosSearchOptions={
            "kind": "vector-diskann",
            "dimensions": 1536,  # Matches OpenAI text-embedding-ada-002
            "similarity": "COS",
            "maxDegree": 32,
            "lBuild": 50
        }
    )
    print("Vector search index created successfully!")
except Exception as e:
    print(f"Vector search index creation failed: {e}")

print("Creating search index...")
# Create text search index
try:
    collection.create_index([("title", "text"), ("description", "text")], name="search_index")
    print("Search index created successfully!")
except Exception as e:
    print(f"Search index creation failed: {e}")

# Run the agent
def run_agent():
    session_id = input("Enter a session ID: ")
    config = {"configurable": {"thread_id": session_id}}

    print("Ask me about books! Type 'quit' to exit.")

    while True:
        user_input = input("Your query: ")
        if user_input.lower() == 'quit':
            break

        # Invoke the graph
        result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config=config)

        # Print the final response
        for message in result["messages"]:
            if isinstance(message, AIMessage) and not message.tool_calls:
                print(f"Answer: {message.content}")

if __name__ == "__main__":
    run_agent()