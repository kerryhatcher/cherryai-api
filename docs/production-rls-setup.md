# Production RLS Entry Gate Setup

Enables Row-Level Security on the production DigitalOcean Postgres database.
Without this setup, the API connects as a superuser and RLS policies are
unconditionally bypassed — all family permission checks function correctly at
the application layer (Capability/scope_sql), but the Postgres-level RLS
backstop (`FORCE ROW LEVEL SECURITY` on family-scoped tables) does nothing.

**Risk if skipped:** A bug that omits a `WHERE` scope filter could leak
cross-family data. RLS is the last line of defense.

---

## Prerequisites

- Admin access to the DO Postgres database (the `cherryai` superuser credential)
- The `APP_DATABASE_URL` must be different from `DATABASE_URL` — the former
  connects as `cherryai_app` (non-superuser), the latter as `cherryai` (superuser)

---

## Step 1 — Create the app role on DO Postgres

Connect to the production database:

```bash
psql "postgresql://cherryai:<superuser-password>@<host>:<port>/cherryai?sslmode=require"
```

Run the production setup script:

```sql
\i scripts/setup-production-app-role.sql
```

This script:
1. Creates the `cherryai_app` role (idempotent)
2. Grants `USAGE` on `public` schema
3. Grants `SELECT, INSERT, UPDATE, DELETE` on all existing tables
4. Sets `ALTER DEFAULT PRIVILEGES` so future migration-created tables
   auto-grant to `cherryai_app`
5. Verifies the role

**Important:** Before running, edit the script to replace `CHANGE_ME_TO_A_STRONG_PASSWORD`
with a strong random password.

---

## Step 2 — Get a strong password

Generate one:

```bash
openssl rand -base64 32
```

Update the SQL script with this password, then run it.

---

## Step 3 — Set APP_DATABASE_URL in DO App Platform

The `.do/app.yaml` already has the `APP_DATABASE_URL` entry. Set its value in
the DO dashboard or via `doctl`:

```bash
doctl apps update <app-id> --spec .do/app.yaml
```

The value should be:

```
postgresql://cherryai_app:<generated-password>@<host>:<port>/cherryai?sslmode=require
```

---

## Step 4 — Deploy

Push to `main` (or trigger a manual deploy). On startup:

1. Migrations run as superuser (`DATABASE_URL`) — creates tables, enables RLS
2. After migrations, the asyncpg pool opens as `cherryai_app` (`APP_DATABASE_URL`)
3. `assert_rls_enforced()` runs and verifies RLS is active on all expected tables
4. Normal operations proceed under the non-superuser role — RLS now enforces

---

## Verify

Check the app logs for the RLS assertion message on startup. You should see:

```
RLS assertion: all registered tables have FORCE RLS enabled
```

If RLS is not enforced, the startup will log a warning and the app continues
with reduced protection (app-layer scoping still works).

---

## Rollback

If the `cherryai_app` role causes issues, set `APP_DATABASE_URL` to the same
value as `DATABASE_URL` (or delete the env var entirely — the code falls back
to `DATABASE_URL`). This makes the API connect as superuser again, bypassing RLS.

---

## Reference

- `docker/initdb/01-create-app-role.sql` — the dev equivalent (same pattern)
- `scripts/setup-production-app-role.sql` — the production script
- `src/cherryai_api/settings.py` — `app_database_url` field with fallback
- `src/cherryai_api/orm.py` — `app_sqlalchemy_url()` uses app role URL
- `src/cherryai_api/db.py` — `build_database(use_app_role=True)` connects as app role
- `tests/test_rls_machinery.py` — demonstrates the same pattern in tests
