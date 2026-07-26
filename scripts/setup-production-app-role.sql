-- ============================================================================
-- Production RLS App Role Setup
-- ============================================================================
-- This script creates the non-superuser application role that the CherryAI
-- API uses at runtime so Postgres Row-Level Security policies take effect.
--
-- Superusers (cherryai) unconditionally bypass RLS. The runtime role
-- (cherryai_app) is NOSUPERUSER NOBYPASSRLS, making RLS actually enforce
-- the per-family, per-module permission model.
--
-- USAGE:
--   1. SSH tunnel or psql directly to your DO Postgres:
--      psql "postgresql://cherryai:<password>@<host>:<port>/cherryai?sslmode=require"
--   2. Run this script:
--      \i scripts/setup-production-app-role.sql
--   3. Update the password below to a strong random value.
--   4. Set APP_DATABASE_URL in the DO app spec to:
--      postgresql://cherryai_app:<new-password>@<host>:<port>/cherryai?sslmode=require
--   5. Redeploy.
-- ============================================================================

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cherryai_app') THEN
        CREATE ROLE cherryai_app LOGIN PASSWORD 'CHANGE_ME_TO_A_STRONG_PASSWORD' NOSUPERUSER NOBYPASSRLS;
        -- IMPORTANT: DO's secret encryption strips special characters from SECRET
        -- type env vars. The password MUST be alphanumeric only (a-zA-Z0-9).
        -- DO NOT use characters like / + = in the password.
    ELSE
        RAISE NOTICE 'Role cherryai_app already exists — skipping creation.';
    END IF;
END
$$;

-- Schema access (CREATE needed for temp tables, Cognee, etc.)
GRANT USAGE ON SCHEMA public TO cherryai_app;
GRANT CREATE ON SCHEMA public TO cherryai_app;

-- Full DML on existing tables
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cherryai_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO cherryai_app;

-- Future tables/sequences (created by migrations, run as superuser) auto-grant
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cherryai_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO cherryai_app;

-- Verify
SELECT rolname, rolcanlogin, rolsuper, rolbypassrls
FROM pg_catalog.pg_roles
WHERE rolname = 'cherryai_app';
