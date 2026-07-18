"""Tests for the web_search Tavily-primary / Brave-fallback logic."""

import cherryai_api.agent as agent_module
from cherryai_api.agent import run_web_search
from cherryai_api.settings import Settings


def _settings(tavily: str = "", brave: str = "") -> Settings:
    return Settings(
        openrouter_api_key="x",
        tavily_api_key=tavily,
        brave_api_key=brave,
    )


async def test_uses_tavily_when_available(monkeypatch) -> None:
    async def fake_tavily(query: str, key: str) -> str:
        return "tavily-result"

    async def fail_brave(query: str, key: str) -> str:  # pragma: no cover
        raise AssertionError("Brave should not be called when Tavily succeeds")

    monkeypatch.setattr(agent_module, "_tavily_search", fake_tavily)
    monkeypatch.setattr(agent_module, "_brave_search", fail_brave)
    result = await run_web_search("hi", _settings(tavily="t", brave="b"))
    assert result == "tavily-result"


async def test_falls_back_to_brave_on_tavily_error(monkeypatch) -> None:
    async def broken_tavily(query: str, key: str) -> str:
        raise RuntimeError("tavily down")

    async def fake_brave(query: str, key: str) -> str:
        return "brave-result"

    monkeypatch.setattr(agent_module, "_tavily_search", broken_tavily)
    monkeypatch.setattr(agent_module, "_brave_search", fake_brave)
    result = await run_web_search("hi", _settings(tavily="t", brave="b"))
    assert result == "brave-result"


async def test_returns_error_string_when_both_fail(monkeypatch) -> None:
    async def broken_tavily(query: str, key: str) -> str:
        raise RuntimeError("tavily down")

    async def broken_brave(query: str, key: str) -> str:
        raise RuntimeError("brave down")

    monkeypatch.setattr(agent_module, "_tavily_search", broken_tavily)
    monkeypatch.setattr(agent_module, "_brave_search", broken_brave)
    result = await run_web_search("hi", _settings(tavily="t", brave="b"))
    assert "web_search failed" in result


async def test_reports_when_no_keys_configured() -> None:
    result = await run_web_search("hi", _settings())
    assert "unavailable" in result
