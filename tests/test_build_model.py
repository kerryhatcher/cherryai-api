"""Tests for chat model construction (Ollama cloud)."""

import pytest
from pydantic_ai.models.openai import OpenAIChatModel

from cherryai_api.agent import build_model
from cherryai_api.settings import Settings


def test_build_model_uses_ollama_cloud_chat_model() -> None:
    settings = Settings(ollama_api_key="x", chat_model="gpt-oss:120b")
    model = build_model(settings)
    assert isinstance(model, OpenAIChatModel)
    assert model.model_name == "gpt-oss:120b"


def test_build_model_requires_ollama_api_key() -> None:
    settings = Settings(ollama_api_key="")
    with pytest.raises(ValueError, match="OLLAMA_API_KEY"):
        build_model(settings)
