# cherryai-api

CherryAI backend — FastAPI-based AI chat service with Cognee-powered memory, web search, and web fetch capabilities.

## What It Is

A production-oriented backend API for the CherryAI chat system. The server hosts a FastAPI web API for chat sessions and messages, a Typer CLI for operational tasks, and a Pydantic AI agent that powers conversation responses with three integrated tools:

- **web_search** — Search the web via Tavily (with Brave fallback on error)
- **web_fetch** — Retrieve and extract text content from web pages
- **search_memory** — Query conversation history and learned facts via Cognee's graph memory

### Architecture Highlights

- **AI Model**: OpenRouter's `openrouter/free` model via Pydantic AI
- **Memory Layer**: Cognee configured for fast session learning using:
  - PostgreSQL (pgvector) for vector embeddings (similarity search)
  - Neo4j for graph-structured knowledge and relationships
  - Local fastembed for embedding generation (no extra API key)
- **Data Storage**: PostgreSQL for chat sessions, messages, and vector storage
- **CLI**: Typer-based command-line interface for admin tasks and smoke testing

## Setup

### Prerequisites

- **Python 3.13+** (managed via `uv`)
- **Docker** and **docker-compose** (for PostgreSQL with pgvector and Neo4j)
- **API Keys**: 
  - `OPENROUTER_API_KEY` — from OpenRouter (free tier available)
  - `TAVILY_API_KEY` — from Tavily for web search (free tier available)
  - `BRAVE_API_KEY` — from Brave Search (fallback, free tier available)

### Environment Configuration

1. **Set up environment variables:**

   Copy `.env` from a working configuration or create one with these required variables:

   ```bash
   # AI Model
   OPENROUTER_API_KEY=your_key_here

   # Web Search (primary and fallback)
   TAVILY_API_KEY=your_key_here
   BRAVE_API_KEY=your_key_here

   # Database (PostgreSQL with pgvector)
   DATABASE_URL=postgresql://cherryai:cherryai_dev@localhost:5442/cherryai

   # Graph Store (Neo4j)
   NEO4J_URI=neo4j://localhost:7484
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=cherryai_dev
   ```

   **Important:** The docker-compose configuration below maps ports `5442` (Postgres) and `7484`/`7697` (Neo4j) to avoid conflicts with system services. Your `.env` must match these port mappings.

2. **Start the database services:**

   ```bash
   docker compose up -d
   ```

   This starts:
   - **PostgreSQL** (image: `pgvector/pgvector`) on `localhost:5442`
   - **Neo4j** (image: `neo4j:5`) on `localhost:7484` (Bolt) and `7697` (HTTPS)

   Verify they're running:
   ```bash
   docker compose ps
   ```

### Installation and Running

1. **Install dependencies:**

   ```bash
   uv sync
   ```

2. **Start the API server:**

   ```bash
   uv run cherryai serve
   ```

   The API listens on `http://localhost:8000`.

3. **Test the setup:**

   ```bash
   # One-shot chat (useful for smoke testing without the UI)
   uv run cherryai chat "What is the capital of France?"

   # List sessions
   uv run cherryai sessions list
   ```

## API Endpoints

All endpoints are prefixed with `/api`.

### Health Check

- **`GET /api/health`**
  - Returns service status and dependency health (database, AI model, search tools)
  - **Response**: `{ "status": "ok", "dependencies": {...} }`

### Sessions Management

- **`GET /api/sessions`**
  - Fetch recent chat sessions (newest first)
  - **Response**: `[ { "id": "...", "created_at": "...", "updated_at": "...", "title": "..." }, ... ]`

- **`POST /api/sessions`**
  - Create a new chat session
  - **Request**: `{ "title": "string (optional)" }`
  - **Response**: `{ "id": "...", "created_at": "...", "updated_at": "...", "title": "..." }`

### Messages and Chat

- **`GET /api/sessions/{session_id}/messages`**
  - Fetch chat history for a session (up to recent limit)
  - **Response**: `[ { "id": "...", "role": "user|assistant", "content": "...", "created_at": "..." }, ... ]`

- **`POST /api/sessions/{session_id}/messages`**
  - Send a user message and stream the assistant response
  - **Request**: `{ "content": "string" }`
  - **Response**: Server-Sent Events (SSE) stream
    - `data: {"token": "partial response"}` (streaming tokens)
    - `data: {"complete": true}` (stream end marker)
  - After the stream completes, the message pair is persisted to the database and learned by Cognee

## CLI Commands

Use `cherryai --help` to see all available commands.

### Common Commands

```bash
# Start the API server (default port 8000)
uv run cherryai serve

# One-shot chat for testing
uv run cherryai chat "Your question here"

# List all sessions
uv run cherryai sessions list

# Help for all commands
uv run cherryai --help
```

## Code Structure

```
src/cherryai_api/
├── __init__.py          # Package initialization
├── settings.py          # Pydantic settings; loads .env variables
├── db.py                # PostgreSQL session/message persistence
├── memory.py            # Cognee integration (embeddings + graph)
├── agent.py             # Pydantic AI agent + three tools
├── api.py               # FastAPI app and route handlers
└── cli.py               # Typer CLI commands
```

### Key Modules

- **`settings.py`**: Loads and validates environment configuration via Pydantic Settings. Env vars must be set *before* importing the memory module.
- **`memory.py`**: Configures Cognee with PostgreSQL (pgvector) + Neo4j. Provides `remember()` and `recall()` methods for storing and retrieving learned facts.
- **`agent.py`**: Defines the Pydantic AI agent with the system prompt and three tools (web_search, web_fetch, search_memory). Uses `OpenRouterModel` and `OpenRouterProvider`.
- **`db.py`**: SQLAlchemy-based async session/message store. Persists all chat turns to Postgres.
- **`api.py`**: FastAPI application. Exposes `/api/health`, `/api/sessions/*`, and `/api/sessions/*/messages` endpoints with SSE streaming support.
- **`cli.py`**: Typer CLI. Commands: `serve`, `chat`, `sessions list`.

## Development

### Linting

Always lint before committing:

```bash
uv run ruff check
```

To auto-fix common issues:

```bash
uv run ruff check --fix
```

### Testing

Run tests to verify core functionality:

```bash
uv run pytest
```

Tests cover:
- Database operations (session and message CRUD)
- Tool fallback logic (Tavily → Brave for web search)
- Integration with Cognee for memory operations

## Deployment Notes

- **Static Hosting**: This API is not meant for static hosting; it requires a running Python environment and access to PostgreSQL/Neo4j.
- **Production**: Before deploying to production, configure:
  - Real PostgreSQL database with backups
  - Real Neo4j instance (or managed Neo4j Aura)
  - Proper authentication and CORS policies (currently allows `http://localhost:5173` for development)
  - Environment-specific settings (database URLs, API keys, etc.)

## Related Documentation

- **[CherryAI Planning Repo](../README.md)** — Project requirements and architecture decisions
- **[cherryai-web README](../cherryai-web/README.md)** — React SPA frontend that consumes this API
- **[Demo Design Spec](../docs/superpowers/specs/2026-07-18-cherryai-demo-design.md)** — Detailed design and execution plan for the working demo

## License

TBD
