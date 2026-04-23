# LangGraph Azure DocumentDB Checkpointer Test

This project demonstrates using LangGraph's MongoDB checkpointer with Azure Cosmos DB's DocumentDB API for persistent agent conversations. It builds an interactive book-finding agent that uses vector search to recommend books based on descriptions.

## Features

- **LangGraph Agent**: A conversational AI agent built with LangGraph for stateful interactions.
- **Persistent Checkpointer**: Uses MongoDB checkpointer to save and resume conversations in Azure DocumentDB.
- **Vector Search**: Leverages DocumentDB's native vector search (cosmosSearch) for semantic book recommendations.
- **Text Search**: Fallback to full-text search for title-based queries.
- **Memory Tools**: Includes tools for saving user preferences.

## Prerequisites

- Python 3.11+
- Azure DocumentDB cluster with M30+ tier
- OpenAI API key

## Setup

1. **Clone or download the project** and navigate to the directory.

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**:
   - Windows: `venv\Scripts\activate`
   - Unix/Mac: `source venv/bin/activate`

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   - Copy `.env.example` to `.env`
   - Fill in your actual values:
     ```
     OPENAI_API_KEY=your_openai_api_key
     DOCUMENTDB_URI=your_documentdb_connection_string
     ```

6. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

- Enter a session ID when prompted (e.g., `my-session`).
- Ask questions like "Find me books about programming" or "Recommend sci-fi books".
- The agent uses vector search for semantic matching and saves conversation state in DocumentDB.
- Type `quit` to exit.
- Restart with the same session ID to resume the conversation.

## Security Notes

- Never commit `.env` files with real credentials.
- Use environment variables or secure secret management for production.
- The `.gitignore` file excludes sensitive files.

## Architecture

- `main.py`: Loads data, sets up vector store and indexes, runs the interactive loop.
- `agent.py`: Defines the LangGraph agent with tools for search and memory.
- Uses LangChain for embeddings and LangGraph for workflow management.
- Vector search via DocumentDB's cosmosSearch for efficient semantic queries.