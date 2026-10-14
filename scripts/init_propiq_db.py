import psycopg2

conn = psycopg2.connect(dbname='postgres', user='postgres', password='postgrespassword', host='localhost', port=5432)
conn.autocommit = True
cur = conn.cursor()
cur.execute("SELECT 1 FROM pg_database WHERE datname='propiq_db'")
exists = cur.fetchone()
if not exists:
    cur.execute("CREATE DATABASE propiq_db")
    print("Created database propiq_db successfully!")
else:
    print("Database propiq_db already exists!")
conn.close()
