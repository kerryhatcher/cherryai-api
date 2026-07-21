"""Application configuration loaded from environment and a project .env file."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """CherryAI API configuration.

    Values come from the environment or a project-local ``.env`` file. Secrets
    are never rendered in ``repr`` output so they cannot leak into logs.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Web search / fetch tools ---
    tavily_api_key: str = Field(default="", repr=False)
    brave_api_key: str = Field(default="", repr=False)

    # --- Ollama cloud (chat agent + feedback AI workflows) ---
    ollama_base_url: str = "https://ollama.com/v1"
    ollama_api_key: str = Field(default="", repr=False)
    chat_model: str = "gpt-oss:120b"
    workflow_triage_model: str = "gpt-oss:20b"
    workflow_investigate_model: str = "gpt-oss:120b"
    workflow_plan_model: str = "kimi-k2.7-code"

    # --- Datastores ---
    database_url: str = Field(
        default="postgresql://cherryai:cherryai_dev@localhost:5432/cherryai",
        repr=False,
    )
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = Field(default="cherryai_dev", repr=False)

    # --- Cognee memory ---
    # Cognee's cognify extraction LLM. Local Ollama by default: OpenRouter's
    # free tier 502s on Cognee's structured-output calls (provider "Stealth"
    # rejects them), so the graph pipeline must not depend on it.
    cognee_llm_endpoint: str = "http://localhost:11434/v1"
    cognee_llm_model: str = "qwen3:8b"
    cognee_llm_api_key: str = Field(default="ollama", repr=False)
    cognee_root_directory: str = "./.cognee"
    cognee_dataset: str = "cherryai_chat_history"
    cognee_session_id: str = "cherryai-chat"
    # Small result set keeps recall focused on the current question.
    cognee_recall_top_k: int = 3

    # --- Chat fact extraction (local Ollama; saves into the permanent Cognee graph) ---
    fact_extraction_enabled: bool = True
    fact_extraction_model: str = "qwen3:8b"
    ollama_local_base_url: str = "http://localhost:11434/v1"

    # --- HTTP server ---
    # Comma-separated list of allowed browser origins.
    cors_origins: str = "http://localhost:5173"

    # --- Logging ---
    # Directory for the JSONL log file (rotated at 1 MB, 7 gzipped rotations
    # kept). Relative paths resolve against the process working directory.
    log_dir: str = "logs"

    # --- Authentication ---
    # Secret for fastapi-users token machinery (reset/verify signing). Any
    # long random string; MUST be overridden in production.
    auth_secret: str = Field(default="dev-secret-change-me", repr=False)
    # False for local http dev; True behind TLS in production.
    auth_cookie_secure: bool = False
    # Auth cookie/token lifetime. 14 days keeps the iOS PWA logged in.
    auth_token_lifetime_seconds: int = 1209600
    # Bootstrap admin for non-interactive deploys (read by migration 0002
    # and `cherryai users bootstrap`). Field names map to env vars
    # CHERRYAI_ADMIN_EMAIL / CHERRYAI_ADMIN_PASSWORD.
    cherryai_admin_email: str = ""
    cherryai_admin_password: str = Field(default="", repr=False)

    # --- Fastmail CalDAV / CardDAV ---
    # Credentials for calendar and contacts integration.
    # Maps to FASTMAIL_USERNAME / FASTMAIL_APP_PASSWORD env vars.
    fastmail_username: str = Field(default="", repr=False)
    fastmail_app_password: str = Field(default="", repr=False)

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def asyncpg_dsn(self) -> str:
        """Return a DSN asyncpg accepts (it rejects the ``+driver`` suffix)."""
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance so the .env is parsed once."""
    return Settings()
