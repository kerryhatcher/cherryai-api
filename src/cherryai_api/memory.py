"""Cognee-backed session memory wired to Postgres, pgvector, and Neo4j.

Cognee reads most of its configuration while it is being imported, so every
environment variable it depends on must be set *before* ``import cognee`` runs
at the bottom of this module. Import this module before anything imports Cognee.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from cherryai_api.settings import get_settings

# On-device embeddings so memory needs no second API key.
EMBEDDING_PROVIDER = "fastembed"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384

_settings = get_settings()

# Cognee's SDK defaults to multi-user authentication meant for its hosted
# server. This demo owns a single local store and never exposes Cognee's API.
_ = os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
# Cognee leaves its telemetry aiohttp session open in short-lived processes;
# disable it before import to avoid the resulting resource-leak warning.
_ = os.environ.setdefault("TELEMETRY_DISABLED", "true")

os.environ["EMBEDDING_PROVIDER"] = EMBEDDING_PROVIDER
os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL
os.environ["EMBEDDING_DIMENSIONS"] = str(EMBEDDING_DIMENSIONS)


def _configure_relational_and_vector() -> None:
    """Point Cognee's relational + pgvector store at the demo Postgres."""
    parsed = urlparse(_settings.asyncpg_dsn)
    host = parsed.hostname or "localhost"
    port = str(parsed.port or 5432)
    username = unquote(parsed.username or "cherryai")
    password = unquote(parsed.password or "cherryai_dev")
    name = parsed.path.lstrip("/") or "cherryai"
    os.environ["DB_PROVIDER"] = "postgres"
    os.environ["DB_HOST"] = host
    os.environ["DB_PORT"] = port
    os.environ["DB_USERNAME"] = username
    os.environ["DB_PASSWORD"] = password
    os.environ["DB_NAME"] = name
    # pgvector reuses the same Postgres; set its credentials explicitly so
    # Cognee does not warn about falling back to the relational configuration.
    os.environ["VECTOR_DB_PROVIDER"] = "pgvector"
    os.environ["VECTOR_DB_HOST"] = host
    os.environ["VECTOR_DB_PORT"] = port
    os.environ["VECTOR_DB_USERNAME"] = username
    os.environ["VECTOR_DB_PASSWORD"] = password
    os.environ["VECTOR_DB_NAME"] = name


def _configure_graph() -> None:
    """Point Cognee's knowledge graph at the demo Neo4j instance."""
    os.environ["GRAPH_DATABASE_PROVIDER"] = "neo4j"
    os.environ["GRAPH_DATABASE_URL"] = _settings.neo4j_uri
    os.environ["GRAPH_DATABASE_USERNAME"] = _settings.neo4j_user
    os.environ["GRAPH_DATABASE_PASSWORD"] = _settings.neo4j_password


def _configure_cognee_llm() -> None:
    """Use OpenRouter as Cognee's structured-output extraction model."""
    os.environ["LLM_PROVIDER"] = "custom"
    os.environ["LLM_MODEL"] = _settings.openrouter_model
    os.environ["LLM_ENDPOINT"] = "https://openrouter.ai/api/v1"
    os.environ["LLM_API_KEY"] = _settings.openrouter_api_key


def _configure_directories() -> None:
    """Keep every Cognee database, cache, and log under ./.cognee/."""
    root = Path(_settings.cognee_root_directory).expanduser().resolve()
    data = root / "data"
    system = root / "system"
    cache = root / "cache"
    logs = root / "logs"
    for directory in (data, system, cache, logs):
        directory.mkdir(parents=True, exist_ok=True)
    os.environ["DATA_ROOT_DIRECTORY"] = str(data)
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system)
    os.environ["CACHE_ROOT_DIRECTORY"] = str(cache)
    os.environ["COGNEE_LOGS_DIR"] = str(logs)


_configure_relational_and_vector()
_configure_graph()
_configure_cognee_llm()
_configure_directories()

import cognee  # noqa: E402  (configuration above must run before this import)
from cognee.modules.data.exceptions.exceptions import (  # noqa: E402
    DatasetNotFoundError,
)


class CogneeMemory:
    """Store chat turns in Cognee and expose fast semantic recall."""

    def __init__(self, dataset: str, session_id: str, top_k: int = 3) -> None:
        self.dataset = dataset
        self.session_id = session_id
        self.top_k = top_k

    async def remember_turn(self, prompt: str, answer: str) -> None:
        """Persist one complete user/assistant turn in Cognee."""
        turn = f"User: {prompt}\nAssistant: {answer}"
        _ = await cognee.remember(
            turn,
            dataset_name=self.dataset,
            session_id=self.session_id,
            # Session memory is immediately searchable and skips the slower
            # knowledge-graph extraction pipeline for chat turns.
            self_improvement=False,
        )

    async def _recall(self, query: str, scope: list[str]) -> str:
        results = await cognee.recall(
            query,
            datasets=[self.dataset],
            session_id=self.session_id,
            scope=scope,
            only_context=True,
            auto_route=False,
            top_k=self.top_k,
        )
        return _format_results(results)

    async def recall(self, query: str) -> str:
        """Return chat and graph memories relevant to a query, as text.

        Prefer session + knowledge-graph scope. Chat turns are stored as session
        memory only (no graph extraction), so when no graph dataset exists Cognee
        raises DatasetNotFoundError; fall back to session-only recall in that case.
        """
        try:
            return await self._recall(query, ["session", "graph"])
        except DatasetNotFoundError:
            return await self._recall(query, ["session"])


def _format_results(results: Sequence[Any]) -> str:
    """Convert Cognee's typed recall responses into tool-friendly text."""
    memories: list[str] = []
    for result in results:
        for field in ("content", "text", "answer", "context"):
            value = getattr(result, field, None)
            if value:
                memories.append(str(value))
                break
        else:
            memories.append(str(result))
    return "\n\n".join(memories) if memories else "No relevant chat history was found."


def build_memory() -> CogneeMemory:
    """Construct the CogneeMemory configured for this demo."""
    return CogneeMemory(
        _settings.cognee_dataset,
        _settings.cognee_session_id,
        top_k=_settings.cognee_recall_top_k,
    )
