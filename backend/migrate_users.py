#for reading the json file
import json
import psycopg2   #POstgre SQL driver for python, this is how python talks to the db 

# DB connection
'''Opens a connection to PostgreSQL (Connect my Python program to the database)

host → where DB is running (your machine)
database → which DB to use
user → DB user (your app_user)
password → its password'''

conn = psycopg2.connect(
    host="localhost",
    database="rbac_db",
    user="app_user",
    password="1234"   # use your actual password
)

cur = conn.cursor()   #Tool to run sql queries

# Load JSON
with open("users.json", "r") as f:       #opens user.json and converts it to python dictionary
    users = json.load(f)

#Loop through users
for emp_id, user in users.items():
    #starts the sql command
    cur.execute("""                      
        INSERT INTO users (
            emp_id, name, password_hash, role,
            failed_attempts, lock_until, lock_count,
            last_failed_login, refresh_token,
            refresh_token_expiry, session_start
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (emp_id) DO NOTHING;
    """, (
        emp_id,
        user.get("name"),
        user.get("password_hash"),
        user.get("role"),
        user.get("failed_attempts", 0),
        user.get("lock_until", 0),
        user.get("lock_count", 0),
        user.get("last_failed_login", 0),
        user.get("refresh_token"),
        user.get("refresh_token_expiry", 0),
        user.get("session_start", 0)
    ))

conn.commit()   #writes all inserts permanently to DB
cur.close()     #close cursor and db connection
conn.close()     

print("Migration complete!")  #tells script is complete