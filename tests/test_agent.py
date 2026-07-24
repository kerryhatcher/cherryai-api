"""Tests for agent streaming behavior.

Reproduces the "Stephanie Hatcher" bug: when the model narrates text
alongside a tool call ("Let me check the wiki:"), the streamed turn must
still run the tool, feed its result back to the model, and finish with the
model's real answer — not end early on the narration.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from cherryai_api.agent import AgentDeps, stream_turn


class _FakeMemory:
    """A stand-in for CogneeMemory: no real Cognee calls."""

    async def recall(self, query: str) -> str:  # pragma: no cover - unused by these tests
        return ""


def _deps() -> AgentDeps:
    return AgentDeps(memory=_FakeMemory(), user_id=uuid.uuid4())


NARRATION = "I don't have anything on that person. Let me check the wiki:"
WIKI_RESULT = "Stephanie Hatcher (/wiki/stephanie-hatcher) Wife of Kerry Hatcher."
FINAL_ANSWER = "Stephanie Hatcher is the wife of Kerry Hatcher."


def _saw_tool_return(messages: list[ModelMessage]) -> bool:
    return any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts)


def _narrate_then_answer(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _saw_tool_return(messages):
        return ModelResponse(parts=[TextPart(FINAL_ANSWER)])
    return ModelResponse(
        parts=[
            TextPart(NARRATION),
            ToolCallPart(tool_name="search_wiki", args={"query": "Stephanie Hatcher"}),
        ]
    )


async def _stream_narrate_then_answer(messages: list[ModelMessage], info: AgentInfo):
    if _saw_tool_return(messages):
        yield FINAL_ANSWER
        return
    yield NARRATION
    yield {1: DeltaToolCall(name="search_wiki", json_args='{"query": "Stephanie Hatcher"}')}


def _build_narrating_agent() -> tuple[Agent[None, str], list[str]]:
    """An agent whose model narrates before calling its only tool."""
    model = FunctionModel(
        function=_narrate_then_answer,
        stream_function=_stream_narrate_then_answer,
    )
    agent: Agent[None, str] = Agent(model)
    tool_calls: list[str] = []

    @agent.tool_plain
    async def search_wiki(query: str) -> str:
        tool_calls.append(query)
        return WIKI_RESULT

    return agent, tool_calls


async def test_stream_turn_answers_from_tool_results_despite_narration():
    agent, tool_calls = _build_narrating_agent()

    events = [event async for event in stream_turn(agent, "who is Stephanie Hatcher", deps=_deps())]

    kind, final = events[-1]
    assert kind == "done"
    assert final == FINAL_ANSWER
    assert tool_calls == ["Stephanie Hatcher"]


def _build_echo_agent(reply: str) -> Agent[None, str]:
    """An agent whose model always answers with ``reply`` verbatim."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(reply)])

    async def stream_respond(messages: list[ModelMessage], info: AgentInfo):
        yield reply

    return Agent(FunctionModel(function=respond, stream_function=stream_respond))


async def _final_payload(agent: Agent[None, str]) -> str:
    events = [event async for event in stream_turn(agent, "who is Stephanie Hatcher", deps=_deps())]
    kind, final = events[-1]
    assert kind == "done"
    return final


async def test_stream_turn_strips_leaked_thought_header():
    agent = _build_echo_agent("thought\nBased on the wiki, Stephanie is Kerry's wife.")
    assert await _final_payload(agent) == "Based on the wiki, Stephanie is Kerry's wife."


async def test_stream_turn_strips_leaked_think_block():
    agent = _build_echo_agent(
        "<think>The user wants the wiki entry.</think>\nStephanie is Kerry's wife."
    )
    assert await _final_payload(agent) == "Stephanie is Kerry's wife."


async def test_stream_turn_keeps_ordinary_answers_untouched():
    agent = _build_echo_agent("Thoughtful gardens need thought and care.")
    assert await _final_payload(agent) == "Thoughtful gardens need thought and care."


class _DatabaseStub:
    """A stand-in for Database: only the `.pool` attribute is read."""

    def __init__(self, pool) -> None:
        self.pool = pool


def _respond_search_wiki(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    if _saw_tool_return(messages):
        return ModelResponse(parts=[TextPart("done")])
    return ModelResponse(parts=[ToolCallPart(tool_name="search_wiki", args={"query": "anything"})])


@pytest.mark.asyncio
async def test_search_wiki_tool_scopes_to_deps_user(pool, make_user, monkeypatch):
    """The wiki tool must pass the deps user id into search_entries."""
    from cherryai_api import agent as agent_mod
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    seen: dict = {}

    async def fake_search(pool_arg, owner_id, query):
        seen["owner_id"] = owner_id
        return []

    monkeypatch.setattr(agent_mod, "search_entries", fake_search)

    user = await make_user("ztest-search-wiki-deps@example.com")
    agent = build_agent(
        Settings(ollama_api_key="x"),
        database=_DatabaseStub(pool),
    )
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    model = FunctionModel(function=_respond_search_wiki)
    await agent.run("who is in the wiki?", deps=deps, model=model)

    assert seen["owner_id"] == deps.user_id


# --- Meal-planning tools (D6) --------------------------------------------------


def _sequential_tool_responder(steps: list[tuple[str, dict]]):
    """A FunctionModel function that plays `steps` one tool call at a time.

    Counts ToolReturnParts already in the message history to decide which
    step is next — stateless, so it works regardless of how many times
    FunctionModel invokes it. Once every step has returned, answers "done".
    """

    def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        completed = sum(
            1 for message in messages for part in message.parts if isinstance(part, ToolReturnPart)
        )
        if completed < len(steps):
            tool_name, args = steps[completed]
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart("done")])

    return _respond


def _tool_returns(result) -> list:
    """Every ToolReturnPart's raw content (dict/list), in call order, from a completed run."""
    return [
        part.content
        for message in result.all_messages()
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


@pytest.mark.asyncio
async def test_meal_tools_registered_on_chat_agent(pool):
    """All 21 meal tools must be registered on the main chat agent by name."""
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    tool_names = set(agent._function_toolset.tools)

    expected = {
        "search_recipes",
        "get_recipe",
        "create_recipe",
        "update_recipe",
        "list_meal_plans",
        "get_meal_plan",
        "create_meal_plan",
        "assign_recipe_to_day",
        "remove_recipe_from_day",
        "mark_meal_consumed",
        "unmark_meal_consumed",
        "generate_shopping_list",
        "list_shopping_lists",
        "get_shopping_list",
        "add_shopping_item",
        "check_off_item",
        "get_pantry",
        "set_pantry_item",
        "list_stores",
        "list_store_products",
        "upsert_store_product",
    }
    assert expected <= tool_names


@pytest.mark.asyncio
async def test_workflow_agents_do_not_have_meal_tools(pool, monkeypatch):
    """Workflow agents are workspace-scoped and must never get meal tools."""
    from cherryai_api import workflows as workflows_mod
    from cherryai_api.settings import Settings

    class _FakeCogneeMemory:
        async def recall(self, query: str) -> str:  # pragma: no cover - unused
            return ""

    agent = workflows_mod.build_investigate_agent(
        Settings(ollama_api_key="x"), database=_DatabaseStub(pool), memory=_FakeCogneeMemory()
    )
    tool_names = set(agent._function_toolset.tools)

    assert "get_pantry" not in tool_names
    assert "search_recipes" not in tool_names
    assert "generate_shopping_list" not in tool_names
    # It should still have its own search tools, proving this isn't an empty toolset.
    assert "search_wiki" in tool_names


@pytest.mark.asyncio
async def test_get_pantry_tool_scopes_to_deps_user(pool, make_user, monkeypatch):
    """The pantry tool must pass the deps user id into list_pantry_items."""
    from cherryai_api import meals as meals_mod
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    seen: dict = {}

    async def fake_list_pantry_items(pool_arg, owner_id):
        seen["owner_id"] = owner_id
        return []

    monkeypatch.setattr(meals_mod, "list_pantry_items", fake_list_pantry_items)

    user = await make_user("ztest-agent-pantry-deps@example.com")
    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    model = FunctionModel(function=_sequential_tool_responder([("get_pantry", {})]))
    await agent.run("what's in my pantry?", deps=deps, model=model)

    assert seen["owner_id"] == deps.user_id


@pytest.mark.asyncio
async def test_meal_tool_reports_unavailable_without_database():
    """Every meal tool checks `database is None` before touching the pool."""
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    agent = build_agent(Settings(ollama_api_key="x"), database=None)
    deps = AgentDeps(memory=_FakeMemory(), user_id=uuid.uuid4())

    model = FunctionModel(function=_sequential_tool_responder([("get_pantry", {})]))
    result = await agent.run("what's in my pantry?", deps=deps, model=model)

    returns = _tool_returns(result)
    assert "unavailable" in returns[0][0]["error"]


@pytest.mark.asyncio
async def test_get_recipe_tool_is_owner_scoped(pool, make_user):
    """get_recipe must find the owner's own recipe but not another user's."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import RecipeCreate, create_recipe
    from cherryai_api.settings import Settings

    alice = await make_user("ztest-agent-recipe-alice@example.com")
    bob = await make_user("ztest-agent-recipe-bob@example.com")
    recipe = await create_recipe(pool, alice["id"], RecipeCreate(name="Ztest Agent Recipe"))

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    model = FunctionModel(
        function=_sequential_tool_responder([("get_recipe", {"recipe_id": str(recipe.id)})])
    )

    alice_deps = AgentDeps(memory=_FakeMemory(), user_id=alice["id"])
    alice_result = await agent.run("show me that recipe", deps=alice_deps, model=model)
    assert _tool_returns(alice_result)[0]["name"] == "Ztest Agent Recipe"

    bob_deps = AgentDeps(memory=_FakeMemory(), user_id=bob["id"])
    bob_result = await agent.run("show me that recipe", deps=bob_deps, model=model)
    assert "No recipe found" in _tool_returns(bob_result)[0]["error"]


@pytest.mark.asyncio
async def test_create_recipe_tool_persists_to_database(pool, make_user):
    """Exercises the create_recipe tool's alias to the module-level create_recipe."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import list_recipes
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-create-recipe@example.com")
    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [
        (
            "create_recipe",
            {
                "name": "Ztest Agent Pasta",
                "ingredients": [{"name": "Ztest Agent Noodles", "quantity": 1.0, "unit": "lb"}],
            },
        )
    ]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("create a pasta recipe", deps=deps, model=model)

    returned = _tool_returns(result)[0]
    assert returned["name"] == "Ztest Agent Pasta"
    assert returned["ingredients"][0]["name"] == "Ztest Agent Noodles"
    recipes = await list_recipes(pool, user["id"])
    assert any(r.name == "Ztest Agent Pasta" for r in recipes)


@pytest.mark.asyncio
async def test_update_recipe_tool_persists_change(pool, make_user):
    """Exercises the update_recipe tool's alias to the module-level update_recipe."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import RecipeCreate, create_recipe, get_recipe
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-update-recipe@example.com")
    recipe = await create_recipe(pool, user["id"], RecipeCreate(name="Ztest Agent Old Name"))

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("update_recipe", {"recipe_id": str(recipe.id), "name": "Ztest Agent New Name"})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("rename that recipe", deps=deps, model=model)

    assert _tool_returns(result)[0]["name"] == "Ztest Agent New Name"
    updated = await get_recipe(pool, user["id"], recipe.id)
    assert updated is not None and updated.name == "Ztest Agent New Name"


@pytest.mark.asyncio
async def test_create_meal_plan_tool_rejects_non_monday(pool, make_user):
    """Exercises the create_meal_plan tool's alias and its Monday validation."""
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-plan-monday@example.com")
    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("create_meal_plan", {"name": "Ztest Agent Plan", "week_start": "2026-07-21"})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("plan my week starting Tuesday", deps=deps, model=model)

    assert "create_meal_plan failed" in _tool_returns(result)[0]["error"]


@pytest.mark.asyncio
async def test_assign_and_get_meal_plan_and_remove_recipe_tools(pool, make_user):
    """Exercises get_meal_plan, assign_recipe_to_day, and remove_recipe_from_day."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import MealPlanCreate, RecipeCreate, create_meal_plan, create_recipe
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-assign@example.com")
    plan = await create_meal_plan(
        pool,
        user["id"],
        MealPlanCreate(name="Ztest Agent Assign Plan", week_start=date(2026, 7, 20)),
    )
    recipe = await create_recipe(pool, user["id"], RecipeCreate(name="Ztest Agent Assign Recipe"))

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [
        (
            "assign_recipe_to_day",
            {
                "plan_id": str(plan.id),
                "day_date": "2026-07-20",
                "recipe_id": str(recipe.id),
                "meal_type": "dinner",
            },
        ),
        ("get_meal_plan", {"plan_id": str(plan.id)}),
    ]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("add that recipe to Monday dinner", deps=deps, model=model)

    returns = _tool_returns(result)
    assign_return, plan_return = returns[0], returns[1]
    assert assign_return["day_date"] == "2026-07-20"
    assert any(r["name"] == "Ztest Agent Assign Recipe" for r in assign_return["recipes"])
    assert any(
        r["name"] == "Ztest Agent Assign Recipe"
        for day in plan_return["days"]
        for r in day["recipes"]
    )

    day_id = assign_return["id"]
    remove_steps = [
        ("remove_recipe_from_day", {"day_id": day_id, "recipe_id": str(recipe.id)}),
    ]
    remove_model = FunctionModel(function=_sequential_tool_responder(remove_steps))
    remove_result = await agent.run("remove it", deps=deps, model=remove_model)
    assert _tool_returns(remove_result)[0]["removed"] is True


@pytest.mark.asyncio
async def test_mark_and_unmark_meal_consumed_tools(pool, make_user):
    """Exercises mark_meal_consumed's pantry-deduction report and unmark_meal_consumed."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import (
        MealPlanCreate,
        MealPlanDayCreate,
        PantryItemCreate,
        RecipeCreate,
        RecipeIngredientCreate,
        add_ingredient,
        add_recipe_to_day,
        create_meal_plan,
        create_recipe,
        upsert_pantry_item,
        upsert_plan_day,
    )
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-consume@example.com")
    plan = await create_meal_plan(
        pool,
        user["id"],
        MealPlanCreate(name="Ztest Agent Consume Plan", week_start=date(2026, 7, 20)),
    )
    recipe = await create_recipe(pool, user["id"], RecipeCreate(name="Ztest Agent Consume Recipe"))
    await add_ingredient(
        pool, recipe.id, RecipeIngredientCreate(name="Ztest Agent Flour", quantity=1.0, unit="cup")
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))
    await add_recipe_to_day(pool, user["id"], day.id, recipe.id)
    await upsert_pantry_item(
        pool, user["id"], PantryItemCreate(name="Ztest Agent Flour", quantity=5.0, unit="cup")
    )

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("mark_meal_consumed", {"day_id": str(day.id)})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("I cooked that meal", deps=deps, model=model)

    consumed = _tool_returns(result)[0]
    assert consumed["day"]["consumed_at"] is not None
    assert consumed["report"][0]["status"] == "deducted"

    unmark_steps = [("unmark_meal_consumed", {"day_id": str(day.id)})]
    unmark_model = FunctionModel(function=_sequential_tool_responder(unmark_steps))
    unmark_result = await agent.run("I didn't actually cook it", deps=deps, model=unmark_model)
    assert _tool_returns(unmark_result)[0]["consumed_at"] is None


@pytest.mark.asyncio
async def test_generate_shopping_list_tool_requires_one_of_plans_or_scope(pool, make_user):
    """Exercises generate_shopping_list's alias and its exactly-one-of validation."""
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-generate@example.com")
    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("generate_shopping_list", {})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("make me a shopping list", deps=deps, model=model)

    assert "generate_shopping_list failed" in _tool_returns(result)[0]["error"]


@pytest.mark.asyncio
async def test_generate_shopping_list_tool_resolves_plan_names_case_insensitively(pool, make_user):
    """`plans` accepts a plan name, matched case-insensitively, not just a UUID."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import (
        MealPlanCreate,
        MealPlanDayCreate,
        RecipeCreate,
        RecipeIngredientCreate,
        add_ingredient,
        add_recipe_to_day,
        create_meal_plan,
        create_recipe,
        upsert_plan_day,
    )
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-generate-name@example.com")
    plan = await create_meal_plan(
        pool,
        user["id"],
        MealPlanCreate(name="Ztest Agent Name Plan", week_start=date(2026, 7, 20)),
    )
    recipe = await create_recipe(pool, user["id"], RecipeCreate(name="Ztest Agent Name Recipe"))
    await add_ingredient(
        pool,
        recipe.id,
        RecipeIngredientCreate(name="Ztest Agent Name Milk", quantity=1.0, unit="gal"),
    )
    day = await upsert_plan_day(pool, plan.id, MealPlanDayCreate(day_date=date(2026, 7, 20)))
    await add_recipe_to_day(pool, user["id"], day.id, recipe.id)

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("generate_shopping_list", {"plans": ["ztest agent name plan"]})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("make a shopping list for my plan by name", deps=deps, model=model)

    slist = _tool_returns(result)[0]
    assert any(item["name"] == "Ztest Agent Name Milk" for item in slist["items"])


@pytest.mark.asyncio
async def test_generate_shopping_list_tool_rejects_unknown_plan_name(pool, make_user):
    from cherryai_api.agent import build_agent
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-generate-unknown@example.com")
    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("generate_shopping_list", {"plans": ["Ztest Nonexistent Plan"]})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("shopping list for a plan that doesn't exist", deps=deps, model=model)

    assert "No meal plan found" in _tool_returns(result)[0]["error"]


@pytest.mark.asyncio
async def test_shopping_list_tools_round_trip(pool, make_user):
    """Exercises list_shopping_lists, get_shopping_list, add_shopping_item, check_off_item."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import ShoppingListCreate, create_shopping_list
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-shopping@example.com")
    slist = await create_shopping_list(
        pool, user["id"], ShoppingListCreate(name="Ztest Agent List")
    )

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [
        ("list_shopping_lists", {}),
        (
            "add_shopping_item",
            {"list_id": str(slist.id), "name": "Ztest Agent Milk", "quantity": 1.0, "unit": "gal"},
        ),
        ("get_shopping_list", {"list_id": str(slist.id)}),
    ]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("show and update my shopping list", deps=deps, model=model)

    returns = _tool_returns(result)
    assert any(ls["name"] == "Ztest Agent List" for ls in returns[0])
    assert returns[1]["name"] == "Ztest Agent Milk"
    item_id = returns[1]["id"]
    assert any(i["name"] == "Ztest Agent Milk" for i in returns[2]["items"])

    check_steps = [("check_off_item", {"item_id": item_id, "purchased": True})]
    check_model = FunctionModel(function=_sequential_tool_responder(check_steps))
    check_result = await agent.run("check off the milk", deps=deps, model=check_model)
    assert _tool_returns(check_result)[0]["purchased"] is True


@pytest.mark.asyncio
async def test_store_tools_round_trip(pool, make_user):
    """Exercises list_stores, list_store_products, and upsert_store_product (both branches)."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import StoreCreate, create_store
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-stores@example.com")
    store = await create_store(pool, user["id"], StoreCreate(name="Ztest Agent Store"))

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [
        ("list_stores", {}),
        ("list_store_products", {"store_id": str(store.id)}),
        (
            "upsert_store_product",
            {
                "store_id": str(store.id),
                "ingredient_name": "Ztest Agent Chicken",
                "product_name": "Chicken Tenders",
                "package_quantity": 5.0,
                "package_unit": "lb",
            },
        ),
    ]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("manage my store", deps=deps, model=model)

    returns = _tool_returns(result)
    assert any(s["name"] == "Ztest Agent Store" for s in returns[0])
    assert returns[1] == []
    assert returns[2]["ingredient_name"] == "Ztest Agent Chicken"
    assert returns[2]["created"] is True

    # Upserting the same ingredient again should update, not duplicate.
    update_steps = [
        (
            "upsert_store_product",
            {
                "store_id": str(store.id),
                "ingredient_name": "Ztest Agent Chicken",
                "product_name": "Chicken Tenders V2",
                "package_quantity": 3.0,
                "package_unit": "lb",
            },
        ),
        ("list_store_products", {"store_id": str(store.id)}),
    ]
    update_model = FunctionModel(function=_sequential_tool_responder(update_steps))
    update_result = await agent.run("update that product", deps=deps, model=update_model)
    update_returns = _tool_returns(update_result)
    assert update_returns[0]["created"] is False
    assert update_returns[0]["product_name"] == "Chicken Tenders V2"
    assert len(update_returns[1]) == 1  # still just one product, updated in place


@pytest.mark.asyncio
async def test_set_pantry_item_tool_writes_to_database(pool, make_user):
    """Exercises the (non-aliased) set_pantry_item tool with a real DB round trip."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import list_pantry_items
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-set-pantry@example.com")
    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("set_pantry_item", {"name": "Ztest Agent Rice", "quantity": 2.0, "unit": "cup"})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("I bought 2 cups of rice", deps=deps, model=model)

    returned = _tool_returns(result)[0]
    assert returned["name"] == "Ztest Agent Rice"
    assert returned["quantity"] == 2.0
    items = await list_pantry_items(pool, user["id"])
    assert any(i.name == "Ztest Agent Rice" and i.quantity == 2.0 for i in items)


@pytest.mark.asyncio
async def test_search_recipes_tool_filters_by_name(pool, make_user):
    """Exercises the search_recipes tool's alias to the module-level list_recipes."""
    from cherryai_api.agent import build_agent
    from cherryai_api.meals import RecipeCreate, create_recipe
    from cherryai_api.settings import Settings

    user = await make_user("ztest-agent-search-recipes@example.com")
    await create_recipe(pool, user["id"], RecipeCreate(name="Ztest Agent Tacos"))
    await create_recipe(pool, user["id"], RecipeCreate(name="Ztest Agent Salad"))

    agent = build_agent(Settings(ollama_api_key="x"), database=_DatabaseStub(pool))
    deps = AgentDeps(memory=_FakeMemory(), user_id=user["id"])

    steps = [("search_recipes", {"query": "Tacos"})]
    model = FunctionModel(function=_sequential_tool_responder(steps))
    result = await agent.run("find my taco recipe", deps=deps, model=model)

    names = {r["name"] for r in _tool_returns(result)[0]}
    assert names == {"Ztest Agent Tacos"}
