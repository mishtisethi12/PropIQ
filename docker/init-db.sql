-- PropIQ Database Initialization Script
CREATE DATABASE propiq_db;
\connect propiq_db;

-- Include PostgreSQL DDL schema
\i /docker-entrypoint-initdb.d/ddl_postgres.sql;
