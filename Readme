# HappyAI Chat

A sophisticated customer support assistant powered by LangChain LangGraph, FastAPI, and Pinecone. This project implements a conversational AI system that can provide information about HappyAI's services, expertise, and projects while maintaining context through vector database integration.

## Features

- **Intelligent Chat Interface**: Conversational AI assistant specialized in HappyAI customer support
- **Vector Database Integration**: Utilizes Pinecone for efficient similarity search and document retrieval
- **Web Search Capability**: Integrates with Tavily for real-time web search functionality
- **FastAPI Backend**: High-performance API with automatic OpenAPI documentation
- **Docker Support**: Containerized deployment for consistency across environments
- **Comprehensive Logging**: Detailed logging system for monitoring and debugging

## Prerequisites

- Python 3.12+
- Docker (for containerized deployment)
- API Keys:
  - OpenAI API key
  - Pinecone API key
  - Tavily API key (for web search functionality)

## Installation

### Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/okoliechykwuka/happyai-chat.git
   cd happyai-chat
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

### Docker Deployment

1. Pull the Docker image:
   ```bash
   docker pull chukypedro15/happyai-chat
   ```

2. Run the container:
   ```bash
   docker run -d -p 8000:8000 \
     -e OPENAI_API_KEY=your_openai_api_key \
     -e PINECONE_API_KEY=your_pinecone_api_key \
     -e TAVILY_API_KEY=your_tavily_api_key \
     chukypedro15/happyai-chat
   ```

## Project Structure

```
happyai-chat/
├── data/
│   └── faq.json
├── src/
│   ├── utils/
│   │   ├── __init__.py
│   │   └── env_setup.py
│   ├── validators/
│   │   ├── agent_validators.py
│   │   └── pinecone_validators.py
│   ├── vector_database/
│   │   ├── utils.py
│   │   └── vector_db.py
│   ├── agent_tools.py
│   ├── agent.py
│   └── models.py
├── Dockerfile
├── main.py
├── requirements.txt
└── README.md
```

## API Endpoints

### Health Check
```
GET /health
```
Returns the current health status of the service.

### Chat Endpoint
```
POST /chat
```
Accepts chat messages and returns AI responses.

Request body:
```json
{
    "message": "string",
    "thread_id": "optional[int]"
}
```

## Usage

### Starting the Server

1. Local development:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

2. Docker:
   ```bash
   docker run -p 8000:8000 yourdockerhub/happyai-chat
   ```

### Making Requests

```python
import requests

url = "http://localhost:8000/chat"
payload = {
    "message": "What services does HappyAI offer?",
    "thread_id": 42
}
response = requests.post(url, json=payload)
print(response.json())
```

## Configuration

The application can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `PINECONE_API_KEY`: Your Pinecone API key
- `TAVILY_API_KEY`: Your Tavily API key

## Development

### Adding New Tools

1. Create a new tool function in `src/agent_tools.py`:
   ```python
   @tool
   def new_tool(parameter: str):
       """Tool description"""
       # Implementation
       return result
   ```

2. Register the tool in `src/agent.py`.

### Extending Vector Database

1. Add new methods to `PineconeManagment` class in `src/vector_database/vector_db.py`.
2. Update validators in `src/validators/pinecone_validators.py` if needed.
