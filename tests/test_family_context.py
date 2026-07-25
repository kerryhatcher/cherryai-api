import uuid

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from cherryai_api.family_context import (
    ACTIVE_FAMILY_COOKIE,
    ACTIVE_FAMILY_HEADER,
    FamilyContextMiddleware,
    active_family_var,
)

FAM = uuid.uuid4()


async def echo(request):
    fam = active_family_var.get()
    return JSONResponse({"family": str(fam) if fam else None})


def _client() -> TestClient:
    app = Starlette(routes=[Route("/", echo)])
    return TestClient(FamilyContextMiddleware(app))


def test_no_cookie_means_personal():
    assert _client().get("/").json() == {"family": None}


def test_cookie_sets_family():
    r = _client().get("/", cookies={ACTIVE_FAMILY_COOKIE: str(FAM)})
    assert r.json() == {"family": str(FAM)}


def test_header_overrides_cookie():
    other = uuid.uuid4()
    r = _client().get(
        "/",
        cookies={ACTIVE_FAMILY_COOKIE: str(FAM)},
        headers={ACTIVE_FAMILY_HEADER: str(other)},
    )
    assert r.json() == {"family": str(other)}


def test_garbage_value_falls_back_to_personal():
    r = _client().get("/", cookies={ACTIVE_FAMILY_COOKIE: "not-a-uuid"})
    assert r.json() == {"family": None}
