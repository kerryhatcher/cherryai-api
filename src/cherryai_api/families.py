"""Families: org-like groups with per-member roles and module permissions.

Models + constants here; service functions and routes are added by later
tasks in this same module (house style: one module per feature).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column

from cherryai_api.orm import Base

FAMILY_ROLE_ORGANIZER = "organizer"
FAMILY_ROLE_ADMIN = "admin"
FAMILY_ROLE_ADULT = "adult"
FAMILY_ROLE_CHILD = "child"
FAMILY_ROLES = (
    FAMILY_ROLE_ORGANIZER,
    FAMILY_ROLE_ADMIN,
    FAMILY_ROLE_ADULT,
    FAMILY_ROLE_CHILD,
)

PERM_NONE = "none"
PERM_VIEW = "view"
PERM_EDIT = "edit"
PERM_LEVELS = (PERM_NONE, PERM_VIEW, PERM_EDIT)

# Matrix defaults applied when a membership is created (spec §2).
DEFAULT_PERMS = {FAMILY_ROLE_ADULT: PERM_EDIT, FAMILY_ROLE_CHILD: PERM_NONE}


class Family(Base):
    __tablename__ = "families"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class FamilyMembership(Base):
    __tablename__ = "family_memberships"
    __table_args__ = (
        UniqueConstraint("family_id", "user_id", name="uq_membership_family_user"),
        # Exactly one organizer per family (spec §2).
        Index(
            "uq_family_organizer",
            "family_id",
            unique=True,
            postgresql_where=text("role = 'organizer'"),
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    family_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("families.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("user.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String, nullable=False)
    perm_wiki: Mapped[str] = mapped_column(String, nullable=False, default=PERM_NONE)
    perm_meals: Mapped[str] = mapped_column(String, nullable=False, default=PERM_NONE)
    perm_planner: Mapped[str] = mapped_column(String, nullable=False, default=PERM_NONE)
    chat_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    web_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
