"""Family planner: shared projects with tasks and subtasks.

This module owns the planner end to end: the pydantic models, the asyncpg data
access helpers, and the FastAPI router mounted under ``/api/planner``.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from enum import StrEnum

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from cherryai_api.auth import current_verified_user
from cherryai_api.users import User

# ------------------------------------------------------------------
# SQL
# ------------------------------------------------------------------

CREATE_PLANNER_TABLES = """
CREATE TABLE IF NOT EXISTS planner_projects (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    color TEXT,
    owner_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS planner_tasks (
    id UUID PRIMARY KEY,
    project_id UUID NOT NULL REFERENCES planner_projects(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'todo',
    assigned_to UUID,
    due_date DATE,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS planner_tasks_project_idx
    ON planner_tasks (project_id, sort_order);

CREATE TABLE IF NOT EXISTS planner_subtasks (
    id UUID PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES planner_tasks(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    completed BOOLEAN NOT NULL DEFAULT false,
    sort_order INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS planner_subtasks_task_idx
    ON planner_subtasks (task_id, sort_order);
"""


# ------------------------------------------------------------------
# Enums
# ------------------------------------------------------------------


class TaskStatus(StrEnum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


# ------------------------------------------------------------------
# Pydantic models
# ------------------------------------------------------------------


class ProjectCreate(BaseModel):
    name: str
    description: str = ""
    color: str | None = None


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    color: str | None = None


class Project(BaseModel):
    id: uuid.UUID
    name: str
    description: str
    color: str | None
    owner_id: uuid.UUID
    created_at: datetime
    updated_at: datetime


class ProjectListItem(BaseModel):
    """Project in list views: no description, includes task counts."""

    id: uuid.UUID
    name: str
    color: str | None
    owner_id: uuid.UUID
    task_total: int
    task_done: int
    created_at: datetime
    updated_at: datetime


class SubtaskCreate(BaseModel):
    title: str


class Subtask(BaseModel):
    id: uuid.UUID
    task_id: uuid.UUID
    title: str
    completed: bool
    sort_order: int
    created_at: datetime


class TaskCreate(BaseModel):
    title: str
    notes: str = ""
    status: TaskStatus = TaskStatus.TODO
    assigned_to: uuid.UUID | None = None
    due_date: date | None = None
    subtasks: list[SubtaskCreate] = []


class TaskUpdate(BaseModel):
    title: str | None = None
    notes: str | None = None
    status: TaskStatus | None = None
    assigned_to: uuid.UUID | None = None
    due_date: date | None = None
    sort_order: int | None = None


class Task(BaseModel):
    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    notes: str
    status: str
    assigned_to: uuid.UUID | None
    due_date: date | None
    sort_order: int
    subtasks: list[Subtask] = []
    created_at: datetime
    updated_at: datetime


class TaskListItem(BaseModel):
    """Task in list views: no notes, includes subtask counts."""

    id: uuid.UUID
    project_id: uuid.UUID
    title: str
    status: str
    assigned_to: uuid.UUID | None
    due_date: date | None
    sort_order: int
    subtask_total: int
    subtask_done: int
    created_at: datetime
    updated_at: datetime


# ------------------------------------------------------------------
# Data access — Projects
# ------------------------------------------------------------------

_PROJECT_COLUMNS = "id, name, description, color, owner_id, created_at, updated_at"
_PROJECT_LIST_COLUMNS = """
    p.id, p.name, p.color, p.owner_id, p.created_at, p.updated_at,
    COALESCE(tc.total, 0) AS task_total,
    COALESCE(tc.done, 0) AS task_done
"""


async def list_projects(pool: asyncpg.Pool, owner_id: uuid.UUID) -> list[ProjectListItem]:
    rows = await pool.fetch(
        f"""
        SELECT {_PROJECT_LIST_COLUMNS}
          FROM planner_projects p
          LEFT JOIN LATERAL (
              SELECT count(*) AS total,
                     count(*) FILTER (WHERE status = 'done') AS done
                FROM planner_tasks
               WHERE project_id = p.id
          ) tc ON true
         WHERE p.owner_id = $1
         ORDER BY p.updated_at DESC
        """,
        owner_id,
    )
    return [ProjectListItem(**dict(row)) for row in rows]


async def get_project(
    pool: asyncpg.Pool, owner_id: uuid.UUID, project_id: uuid.UUID
) -> Project | None:
    row = await pool.fetchrow(
        f"SELECT {_PROJECT_COLUMNS} FROM planner_projects WHERE id = $1 AND owner_id = $2",
        project_id,
        owner_id,
    )
    return Project(**dict(row)) if row else None


async def create_project(pool: asyncpg.Pool, owner_id: uuid.UUID, data: ProjectCreate) -> Project:
    name = data.name.strip()
    if not name:
        raise ValueError("Project name must not be empty")
    row = await pool.fetchrow(
        f"INSERT INTO planner_projects (id, name, description, color, owner_id) "
        f"VALUES ($1, $2, $3, $4, $5) RETURNING {_PROJECT_COLUMNS}",
        uuid.uuid4(),
        name,
        data.description,
        data.color,
        owner_id,
    )
    return Project(**dict(row))


async def update_project(
    pool: asyncpg.Pool, owner_id: uuid.UUID, project_id: uuid.UUID, data: ProjectUpdate
) -> Project | None:
    name = data.name.strip() if data.name is not None else None
    if data.name is not None and not name:
        raise ValueError("Project name must not be empty")
    row = await pool.fetchrow(
        f"UPDATE planner_projects SET "
        f"name = COALESCE($3, name), "
        f"description = COALESCE($4, description), "
        f"color = COALESCE($5, color), "
        f"updated_at = now() "
        f"WHERE id = $1 AND owner_id = $2 RETURNING {_PROJECT_COLUMNS}",
        project_id,
        owner_id,
        name,
        data.description,
        data.color,
    )
    return Project(**dict(row)) if row else None


async def delete_project(pool: asyncpg.Pool, owner_id: uuid.UUID, project_id: uuid.UUID) -> bool:
    result = await pool.execute(
        "DELETE FROM planner_projects WHERE id = $1 AND owner_id = $2", project_id, owner_id
    )
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Tasks
# ------------------------------------------------------------------

_TASK_COLUMNS = (
    "id, project_id, title, notes, status, "
    "assigned_to, due_date, sort_order, created_at, updated_at"
)
_TASK_COLUMNS_STR = " ".join(_TASK_COLUMNS)
_TASK_LIST_COLUMNS = f"""
    t.{_TASK_COLUMNS_STR},
    COALESCE(sc.total, 0) AS subtask_total,
    COALESCE(sc.done, 0) AS subtask_done
"""


async def list_tasks(pool: asyncpg.Pool, project_id: uuid.UUID) -> list[TaskListItem]:
    rows = await pool.fetch(
        f"""
        SELECT {_TASK_LIST_COLUMNS}
          FROM planner_tasks t
          LEFT JOIN LATERAL (
              SELECT count(*) AS total,
                     count(*) FILTER (WHERE completed) AS done
                FROM planner_subtasks
               WHERE task_id = t.id
          ) sc ON true
         WHERE t.project_id = $1
         ORDER BY t.sort_order, t.created_at
        """,
        project_id,
    )
    return [TaskListItem(**dict(row)) for row in rows]


async def get_task(pool: asyncpg.Pool, task_id: uuid.UUID) -> Task | None:
    row = await pool.fetchrow(
        f"SELECT {_TASK_COLUMNS_STR} FROM planner_tasks WHERE id = $1", task_id
    )
    if row is None:
        return None
    task = dict(row)
    subtask_rows = await pool.fetch(
        "SELECT id, task_id, title, completed, sort_order, created_at "
        "FROM planner_subtasks WHERE task_id = $1 ORDER BY sort_order, created_at",
        task_id,
    )
    task["subtasks"] = [Subtask(**dict(sr)) for sr in subtask_rows]
    return Task(**task)


async def create_task(pool: asyncpg.Pool, project_id: uuid.UUID, data: TaskCreate) -> Task:
    title = data.title.strip()
    if not title:
        raise ValueError("Task title must not be empty")

    # Get the next sort_order
    max_order = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), -1) FROM planner_tasks WHERE project_id = $1",
        project_id,
    )
    sort_order = (max_order or -1) + 1

    async with pool.acquire() as conn:
        async with conn.transaction():
            row = await conn.fetchrow(
                f"INSERT INTO planner_tasks "
                f"(id, project_id, title, notes, status, assigned_to, due_date, sort_order) "
                f"VALUES ($1, $2, $3, $4, $5, $6, $7, $8) RETURNING {_TASK_COLUMNS_STR}",
                uuid.uuid4(),
                project_id,
                title,
                data.notes,
                data.status.value,
                data.assigned_to,
                data.due_date,
                sort_order,
            )
            task = dict(row)
            subtasks: list[Subtask] = []
            for i, st in enumerate(data.subtasks):
                sr = await conn.fetchrow(
                    "INSERT INTO planner_subtasks (id, task_id, title, sort_order) "
                    "VALUES ($1, $2, $3, $4) "
                    "RETURNING id, task_id, title, completed, sort_order, created_at",
                    uuid.uuid4(),
                    task["id"],
                    st.title.strip(),
                    i,
                )
                subtasks.append(Subtask(**dict(sr)))
            task["subtasks"] = subtasks
    return Task(**task)


async def update_task(pool: asyncpg.Pool, task_id: uuid.UUID, data: TaskUpdate) -> Task | None:
    title = data.title.strip() if data.title is not None else None
    if data.title is not None and not title:
        raise ValueError("Task title must not be empty")

    row = await pool.fetchrow(
        f"UPDATE planner_tasks SET "
        f"title = COALESCE($2, title), "
        f"notes = COALESCE($3, notes), "
        f"status = COALESCE($4, status), "
        f"assigned_to = COALESCE($5, assigned_to), "
        f"due_date = COALESCE($6, due_date), "
        f"sort_order = COALESCE($7, sort_order), "
        f"updated_at = now() "
        f"WHERE id = $1 RETURNING {_TASK_COLUMNS_STR}",
        task_id,
        title,
        data.notes,
        data.status.value if data.status else None,
        data.assigned_to,
        data.due_date,
        data.sort_order,
    )
    if row is None:
        return None
    task = dict(row)
    subtask_rows = await pool.fetch(
        "SELECT id, task_id, title, completed, sort_order, created_at "
        "FROM planner_subtasks WHERE task_id = $1 ORDER BY sort_order, created_at",
        task_id,
    )
    task["subtasks"] = [Subtask(**dict(sr)) for sr in subtask_rows]
    return Task(**task)


async def delete_task(pool: asyncpg.Pool, task_id: uuid.UUID) -> bool:
    result = await pool.execute("DELETE FROM planner_tasks WHERE id = $1", task_id)
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — Subtasks
# ------------------------------------------------------------------


async def add_subtask(pool: asyncpg.Pool, task_id: uuid.UUID, data: SubtaskCreate) -> Subtask:
    title = data.title.strip()
    if not title:
        raise ValueError("Subtask title must not be empty")
    max_order = await pool.fetchval(
        "SELECT COALESCE(MAX(sort_order), -1) FROM planner_subtasks WHERE task_id = $1",
        task_id,
    )
    sort_order = (max_order or -1) + 1
    row = await pool.fetchrow(
        "INSERT INTO planner_subtasks (id, task_id, title, sort_order) "
        "VALUES ($1, $2, $3, $4) "
        "RETURNING id, task_id, title, completed, sort_order, created_at",
        uuid.uuid4(),
        task_id,
        title,
        sort_order,
    )
    return Subtask(**dict(row))


async def update_subtask(
    pool: asyncpg.Pool, subtask_id: uuid.UUID, title: str | None, completed: bool | None
) -> Subtask | None:
    row = await pool.fetchrow(
        "UPDATE planner_subtasks SET "
        "title = COALESCE($2, title), "
        "completed = COALESCE($3, completed) "
        "WHERE id = $1 "
        "RETURNING id, task_id, title, completed, sort_order, created_at",
        subtask_id,
        title.strip() if title else None,
        completed,
    )
    return Subtask(**dict(row)) if row else None


async def delete_subtask(pool: asyncpg.Pool, subtask_id: uuid.UUID) -> bool:
    result = await pool.execute("DELETE FROM planner_subtasks WHERE id = $1", subtask_id)
    return result.endswith("1")


# ------------------------------------------------------------------
# Data access — User list (for assignment picker)
# ------------------------------------------------------------------


class PlannerUser(BaseModel):
    id: uuid.UUID
    display_name: str
    email: str


async def list_planner_users(pool: asyncpg.Pool) -> list[PlannerUser]:
    """Return all active, verified users for the assignment picker."""
    rows = await pool.fetch(
        "SELECT id, display_name, email FROM users WHERE is_active AND is_verified "
        "ORDER BY display_name"
    )
    return [PlannerUser(**dict(row)) for row in rows]


# ------------------------------------------------------------------
# Router
# ------------------------------------------------------------------

router = APIRouter(prefix="/api/planner", tags=["planner"])


def _pool(request: Request) -> asyncpg.Pool:
    return request.app.state.db.pool


# -- Users (for assignment picker) --


@router.get("/users")
async def get_planner_users(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    users = await list_planner_users(_pool(request))
    return [u.model_dump(mode="json") for u in users]


# -- Projects --


@router.get("/projects")
async def list_planner_projects(
    request: Request,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    projects = await list_projects(_pool(request), user.id)
    return [p.model_dump(mode="json") for p in projects]


@router.post("/projects", status_code=201)
async def create_planner_project(
    request: Request,
    body: ProjectCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        project = await create_project(_pool(request), user.id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return project.model_dump(mode="json")


@router.get("/projects/{project_id}")
async def get_planner_project(
    request: Request,
    project_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    project = await get_project(_pool(request), user.id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump(mode="json")


@router.put("/projects/{project_id}")
async def update_planner_project(
    request: Request,
    project_id: uuid.UUID,
    body: ProjectUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    try:
        project = await update_project(_pool(request), user.id, project_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project.model_dump(mode="json")


@router.delete("/projects/{project_id}", status_code=204)
async def delete_planner_project(
    request: Request,
    project_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_project(_pool(request), user.id, project_id):
        raise HTTPException(status_code=404, detail="Project not found")


# -- Tasks --


@router.get("/projects/{project_id}/tasks")
async def list_planner_tasks(
    request: Request,
    project_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> list[dict]:
    # Verify project ownership
    project = await get_project(_pool(request), user.id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    tasks = await list_tasks(_pool(request), project_id)
    return [t.model_dump(mode="json") for t in tasks]


@router.post("/projects/{project_id}/tasks", status_code=201)
async def create_planner_task(
    request: Request,
    project_id: uuid.UUID,
    body: TaskCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    project = await get_project(_pool(request), user.id, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        task = await create_task(_pool(request), project_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return task.model_dump(mode="json")


@router.get("/tasks/{task_id}")
async def get_planner_task(
    request: Request,
    task_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    task = await get_task(_pool(request), task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    # Verify project ownership
    project = await get_project(_pool(request), user.id, task.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task.model_dump(mode="json")


@router.put("/tasks/{task_id}")
async def update_planner_task(
    request: Request,
    task_id: uuid.UUID,
    body: TaskUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    task = await get_task(_pool(request), task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    project = await get_project(_pool(request), user.id, task.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        updated = await update_task(_pool(request), task_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return updated.model_dump(mode="json")


@router.delete("/tasks/{task_id}", status_code=204)
async def delete_planner_task(
    request: Request,
    task_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    task = await get_task(_pool(request), task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    project = await get_project(_pool(request), user.id, task.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Task not found")
    if not await delete_task(_pool(request), task_id):
        raise HTTPException(status_code=404, detail="Task not found")


# -- Subtasks --


@router.post("/tasks/{task_id}/subtasks", status_code=201)
async def create_planner_subtask(
    request: Request,
    task_id: uuid.UUID,
    body: SubtaskCreate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    task = await get_task(_pool(request), task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    project = await get_project(_pool(request), user.id, task.project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        subtask = await add_subtask(_pool(request), task_id, body)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return subtask.model_dump(mode="json")


class SubtaskUpdate(BaseModel):
    title: str | None = None
    completed: bool | None = None


@router.put("/subtasks/{subtask_id}")
async def update_planner_subtask(
    request: Request,
    subtask_id: uuid.UUID,
    body: SubtaskUpdate,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> dict:
    subtask = await update_subtask(_pool(request), subtask_id, body.title, body.completed)
    if subtask is None:
        raise HTTPException(status_code=404, detail="Subtask not found")
    return subtask.model_dump(mode="json")


@router.delete("/subtasks/{subtask_id}", status_code=204)
async def delete_planner_subtask(
    request: Request,
    subtask_id: uuid.UUID,
    user: User = Depends(current_verified_user),  # noqa: B008
) -> None:
    if not await delete_subtask(_pool(request), subtask_id):
        raise HTTPException(status_code=404, detail="Subtask not found")
