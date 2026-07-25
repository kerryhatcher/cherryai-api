"""Exhaustive unit tests for the authz scope matrix. No DB needed."""

import itertools
import uuid

import pytest

from cherryai_api.authz import (
    MODULES,
    SCOPE_ADULT_WIKI,
    Capability,
    build_scopes,
    require_permission,
)
from cherryai_api.families import (
    FAMILY_ROLE_ADMIN,
    FAMILY_ROLE_ADULT,
    FAMILY_ROLE_CHILD,
    FAMILY_ROLE_ORGANIZER,
    PERM_EDIT,
    PERM_LEVELS,
    PERM_NONE,
    PERM_VIEW,
)

ALL_EDIT = {m: PERM_EDIT for m in MODULES}


def scopes(role, perms=None, **kw):
    defaults = dict(chat_enabled=True, web_enabled=True, is_child_anywhere=False)
    defaults.update(kw)
    return build_scopes(role=role, perms=perms or ALL_EDIT, **defaults)


def test_personal_context_grants_all_modules():
    s = scopes(None, perms={})
    for m in MODULES:
        assert f"{m}:read" in s and f"{m}:write" in s
    assert "family:manage" not in s


def test_organizer_and_admin_get_full_modules_and_manage():
    for role in (FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN):
        s = scopes(role, perms={m: PERM_NONE for m in MODULES})  # matrix ignored
        for m in MODULES:
            assert f"{m}:read" in s and f"{m}:write" in s
        assert "family:manage" in s
    assert "family:own" in scopes(FAMILY_ROLE_ORGANIZER)
    assert "family:own" not in scopes(FAMILY_ROLE_ADMIN)


def test_matrix_levels_exhaustively_for_adult_and_child():
    for role in (FAMILY_ROLE_ADULT, FAMILY_ROLE_CHILD):
        for combo in itertools.product(PERM_LEVELS, repeat=len(MODULES)):
            perms = dict(zip(MODULES, combo, strict=False))
            s = scopes(role, perms=perms)
            for m, level in perms.items():
                assert (f"{m}:read" in s) == (level in (PERM_VIEW, PERM_EDIT))
                assert (f"{m}:write" in s) == (level == PERM_EDIT)
            assert "family:manage" not in s


def test_child_never_sees_adult_wiki_or_fastmail():
    s = scopes(FAMILY_ROLE_CHILD, perms=ALL_EDIT, is_child_anywhere=True)
    assert SCOPE_ADULT_WIKI not in s
    for scope in ("calendar", "contacts", "email"):
        assert scope not in s


def test_adult_roles_see_adult_wiki():
    for role in (None, FAMILY_ROLE_ORGANIZER, FAMILY_ROLE_ADMIN, FAMILY_ROLE_ADULT):
        assert SCOPE_ADULT_WIKI in scopes(role, perms=ALL_EDIT)


def test_child_anywhere_strips_fastmail_in_every_context():
    for role in (None, FAMILY_ROLE_ADULT):
        s = scopes(role, is_child_anywhere=True)
        for scope in ("calendar", "contacts", "email"):
            assert scope not in s


def test_gates_control_chat_and_web():
    assert "chat" not in scopes(FAMILY_ROLE_CHILD, chat_enabled=False)
    assert "web" not in scopes(FAMILY_ROLE_CHILD, web_enabled=False)
    assert "chat" in scopes(FAMILY_ROLE_CHILD, chat_enabled=True)


def test_capability_has():
    cap = Capability(uuid.uuid4(), None, None, frozenset({"wiki:read"}))
    assert cap.has("wiki:read") and not cap.has("wiki:write")


def test_require_permission_rejects_invalid_level():
    with pytest.raises(ValueError, match="level must be one of"):
        require_permission("wiki", "banana")
