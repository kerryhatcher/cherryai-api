import uuid

from cherryai_api.authz import (
    SCOPE_ADULT_WIKI,
    Capability,
    scope_sql,
    wiki_visibility_sql,
)

UID, FID = uuid.uuid4(), uuid.uuid4()


def cap(family=None, scopes=frozenset()):
    return Capability(UID, family, "child" if family else None, frozenset(scopes))


def test_personal_scope_sql():
    clause, params = scope_sql(cap())
    assert clause == "(family_id IS NULL AND owner_id = $1)"
    assert params == [UID]


def test_family_scope_sql_with_alias_and_offset():
    clause, params = scope_sql(cap(family=FID), alias="w", start=3)
    assert clause == "(w.family_id = $3)"
    assert params == [FID]


def test_wiki_visibility_adds_audience_for_child():
    clause, params = wiki_visibility_sql(cap(family=FID))
    assert clause == "((family_id = $1) AND audience = 'family')"
    assert params == [FID]


def test_wiki_visibility_unchanged_for_adult_scope():
    c = cap(family=FID, scopes={SCOPE_ADULT_WIKI})
    clause, _ = wiki_visibility_sql(c)
    assert "audience" not in clause


def test_wiki_visibility_personal_ignores_audience():
    clause, _ = wiki_visibility_sql(cap())
    assert "audience" not in clause
