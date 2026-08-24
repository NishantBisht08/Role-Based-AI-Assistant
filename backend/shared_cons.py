import os
from psycopg2 import pool

DATABASE_URL = os.environ.get("DATABASE_URL")

if DATABASE_URL:
    # Production pool (Render/Neon)
    connection_pool = pool.ThreadedConnectionPool(
        5,   # min connections
        20,  # max connections
        dsn=DATABASE_URL,
        # --- NEW: TCP KEEPALIVES ---
        keepalives=1,             # 1 = turn on keepalives
        keepalives_idle=30,       # Wait 30 seconds of idle time before sending a heartbeat
        keepalives_interval=10,   # Send a heartbeat every 10 seconds after that
        keepalives_count=5        # Drop the connection if 5 heartbeats fail in a row
    )
    
    
else:
    # Local development pool
    connection_pool = pool.ThreadedConnectionPool(
        5,   # min connections
        20,  # max connections
        host="localhost",
        database="rbac_db",
        user="app_user",
        password="1234",
        # Keepalives are good practice locally too!
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5
    )
