from psycopg2 import pool

# Create a connection pool
connection_pool = pool.SimpleConnectionPool(
    5,   # min connections
    20,  # max connections
    host="localhost",
    database="rbac_db",
    user="app_user",
    password="1234"
)
