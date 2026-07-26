-- Creates the non-superuser application role for RLS enforcement.
-- The default POSTGRES_USER (cherryai) is a superuser, which Postgres
-- unconditionally exempts from row-level security. This role is what the
-- API connects as at runtime so RLS policies actually take effect.
--
-- Migrations still run as the superuser (cherryai) via DATABASE_URL;
-- APP_DATABASE_URL points here for normal operations.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'cherryai_app') THEN
        CREATE ROLE cherryai_app LOGIN PASSWORD 'cherryai_app_dev' NOSUPERUSER NOBYPASSRLS;
    END IF;
END
$$;

-- Grant schema access so the role can interact with tables.
GRANT USAGE ON SCHEMA public TO cherryai_app;

-- Grant full DML on all existing tables. New tables created by migrations
-- (as superuser) need ALTER DEFAULT PRIVILEGES below to auto-grant.
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO cherryai_app;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO cherryai_app;

-- Future tables/sequences created by the superuser will auto-grant to the app role.
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO cherryai_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE ON SEQUENCES TO cherryai_app;
