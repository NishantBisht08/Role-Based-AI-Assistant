import os
from psycopg2 import pool

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    connection_pool = pool.SimpleConnectionPool(
        5,
        20,
        dsn=DATABASE_URL
    )
else:
    connection_pool = pool.SimpleConnectionPool(
        5,
        20,
        host="localhost",
        database="rbac_db",
        user="app_user",
        password="1234"
    )