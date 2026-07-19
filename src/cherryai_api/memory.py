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
    """Point Cognee's structured-output extraction model at local Ollama.

    OpenRouter's free tier cannot serve Cognee's cognify pipeline (its
    structured-output calls 502 at the provider), so extraction runs on the
    local instance instead.
    """
    # Cognee's native ollama provider wraps an OpenAI-compatible client, so
    # the endpoint keeps its /v1 suffix and the model name stays unprefixed
    # (the "custom" provider would require a litellm "openai/" model prefix).
    os.environ["LLM_PROVIDER"] = "ollama"
    os.environ["LLM_MODEL"] = _settings.cognee_llm_model
    os.environ["LLM_ENDPOINT"] = _settings.cognee_llm_endpoint
    os.environ["LLM_API_KEY"] = _settings.cognee_llm_api_key


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
from cognee.modules.users.exceptions.exceptions import (  # noqa: E402
    PermissionDeniedError,
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

    async def remember_fact(self, fact: str) -> None:
        """Persist a durable fact into the permanent knowledge graph.

        Unlike ``remember_turn``, no ``session_id`` is passed: Cognee runs
        the full add + cognify pipeline into the durable graph dataset
        rather than the fast session-only path used for chat turns.
        """
        result = await cognee.remember(
            fact,
            dataset_name=self.dataset,
            node_set=["user_facts"],
        )
        # remember() reports pipeline failures on the result rather than
        # raising, so a broken extraction LLM would otherwise be invisible.
        status = getattr(result, "status", None)
        if status == "errored":
            raise RuntimeError(
                f"Cognee remember() failed: {getattr(result, 'error', 'unknown error')}"
            )

    async def recall_facts(self, query: str) -> str:
        """Recall existing durable facts similar to a candidate fact.

        Facts live in the graph dataset only (no session_id), so this
        queries that scope alone. Before the very first fact is ever saved,
        the dataset does not exist yet, and Cognee raises
        ``DatasetNotFoundError`` or, if backend access control has not
        finished provisioning it, ``PermissionDeniedError`` — both mean "no
        similar facts".
        """
        try:
            results = await cognee.recall(
                query,
                datasets=[self.dataset],
                scope=["graph"],
                only_context=True,
                auto_route=False,
                top_k=self.top_k,
            )
        except (DatasetNotFoundError, PermissionDeniedError):
            return _NO_SIMILAR_FACTS
        return _format_results(results, empty_message=_NO_SIMILAR_FACTS)

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


_NO_SIMILAR_FACTS = "No similar facts were found."


def _format_results(
    results: Sequence[Any], empty_message: str = "No relevant chat history was found."
) -> str:
    """Convert Cognee's typed recall responses into tool-friendly text."""
    memories: list[str] = []
    for result in results:
        for field in ("content", "text", "answer", "context"):
            value = getattr(result, field, None)
            if value:
                memories.append(str(value))
                break
        # Results with no usable text (e.g. an empty graph completion) are
        # skipped rather than rendered as a raw object repr.
    return "\n\n".join(memories) if memories else empty_message


def build_memory(dataset: str | None = None, session_id: str | None = None) -> CogneeMemory:
    """Construct a CogneeMemory; None falls back to the configured defaults."""
    return CogneeMemory(
        dataset or _settings.cognee_dataset,
        session_id or _settings.cognee_session_id,
        top_k=_settings.cognee_recall_top_k,
    )
