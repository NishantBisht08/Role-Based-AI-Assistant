import psycopg2
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

conn = psycopg2.connect(host="localhost", database="rbac_db", user="app_user", password="1234")
cur = conn.cursor()

users = [
    ("sid",     "Sid",     "sid123",     "finance"),
    ("nishant", "Nishant", "nishant123", "engineering"),
    ("khushi",  "Khushi",  "khushi123",  "hr"),
    ("kavya",   "Kavya",   "kavya123",   "marketing"),
    ("arjun",   "Arjun",   "arjun123",   "employee"),
    ("leo",     "Leo",     "leo123",     "c-level"),
    ("admin",   "System Admin", "admin123", "admin"),
]

for emp_id, name, password, role in users:
    hashed = pwd_context.hash(password)
    cur.execute("""
        INSERT INTO users (emp_id, name, password_hash, role, failed_attempts, lock_until, lock_count, last_failed_login, refresh_token, refresh_token_expiry, session_start)
        VALUES (%s, %s, %s, %s, 0, 0, 0, 0, '', 0, 0)
        ON CONFLICT (emp_id) DO NOTHING
    """, (emp_id, name, hashed, role))
    print(f"Inserted {emp_id} ({role}) with password: {password}")

conn.commit()
cur.close()
conn.close()
print("\nAll users seeded successfully!")
