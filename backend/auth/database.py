from backend.shared_cons import connection_pool

from .jwt_handler import create_access_token
from .config import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    ABSOLUTE_SESSION_EXPIRE_DAYS,
)

import hashlib
import secrets
import time


## This function fetches user data from PostgreSQL and returns it as a dictionary
'''# THis function is the bridge between the api and the db
# Called from multiple endpoints (/login, /ask) and auth functions
# Returns data as a dictionary (username → user details)'''

def get_user(emp_id):
    # Get an available connection from the connection pool
    conn = connection_pool.getconn()
    cur = None
    row = None
    bad_connection = False

    try:
        # Create cursor to execute SQL queries
        cur = conn.cursor()

        # Fetch the user from PostgreSQL
        cur.execute(
            "SELECT * FROM users WHERE emp_id = %s",
            (emp_id,)
        )

        # emp_id is unique, so only one row is expected
        row = cur.fetchone()

    except Exception:
        # The connection may be broken, so do not reuse it
        bad_connection = True
        raise

    finally:
        # Safely close the cursor if it was created
        if cur:
            cur.close()

        # If connection is bad, close/discard it.
        # Otherwise, return it to the pool for reuse.
        connection_pool.putconn(conn, close=bad_connection)

    # User not found
    if not row:
        return None

    # Convert database row into a dictionary
    return {
        "emp_id": row[0],
        "name": row[1],
        "password_hash": row[2],
        "role": row[3],
        "failed_attempts": row[4],
        "lock_until": row[5],
        "lock_count": row[6],
        "last_failed_login": row[7],
        "refresh_token": row[8],
        "refresh_token_expiry": row[9],
        "session_start": row[10]
    }



# Update specific user fields in PostgreSQL database
# Called after modifying user data (login, failed attempts, tokens, etc.)
def update_user(emp_id, user):
    # Connect to PostgreSQL database
    conn = connection_pool.getconn()
    cur = None
    bad_connection = False

    try:
        # Create cursor to execute SQL queries
        cur = conn.cursor()

        # Update user data in DB using SQL UPDATE query
        cur.execute("""
            UPDATE users SET
                failed_attempts = %s,
                lock_until = %s,
                lock_count = %s,
                last_failed_login = %s,
                refresh_token = %s,
                refresh_token_expiry = %s,
                session_start = %s,
                password_hash = %s
            WHERE emp_id = %s
        """, (
            user["failed_attempts"],
            user["lock_until"],
            user["lock_count"],
            user["last_failed_login"],
            user.get("refresh_token"),
            user.get("refresh_token_expiry"),
            user.get("session_start"),
            user["password_hash"],
            emp_id
        ))

        # Save changes permanently in database
        conn.commit()

    except Exception:
        # Mark connection as bad so it is not reused
        bad_connection = True

        # Try to rollback the failed transaction
        try:
            conn.rollback()
        except Exception:
            pass

        # Re-raise the original error
        raise

    finally:
        # Safely close the cursor and return/discard the connection
        if cur:
            cur.close()

        connection_pool.putconn(conn, close=bad_connection)




# inserts a new user row in the database
def db_create_user(emp_id, name, role):
    # Create new user WITHOUT password
    conn = connection_pool.getconn()
    cur = None
    bad_connection = False

    try:
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO users (
                emp_id, name, password_hash, role,
                failed_attempts, lock_until, lock_count,
                last_failed_login, refresh_token,
                refresh_token_expiry, session_start
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            emp_id,
            name,
            "",
            role,
            0,
            0,
            0,
            0,
            "",
            0,
            0
        ))

        conn.commit()

    except Exception:
        # Mark connection as bad so it is not reused
        bad_connection = True

        # Try to rollback the failed transaction
        try:
            conn.rollback()
        except Exception:
            pass

        # Re-raise the original error
        raise

    finally:
        # Safely close the cursor and return/discard the connection
        if cur:
            cur.close()

        connection_pool.putconn(conn, close=bad_connection)




# updating refresh token information of the user in the database when his token is rotated
def db_rotate_refresh_token(refresh_token_hash):

    current_time = time.time()
    conn = connection_pool.getconn()
    cur = None
    bad_connection = False

    try:
        cur = conn.cursor()

        # Lock the matching row until token rotation is complete.
        # fetching refresh token information of user
        cur.execute(
            """
            SELECT emp_id, role, lock_until,
                   refresh_token_expiry, session_start
            FROM users
            WHERE refresh_token = %s
            FOR UPDATE
            """,
            (refresh_token_hash,)
        )

        row = cur.fetchone()

        if not row:
            conn.rollback()
            return None

        (
            emp_id,
            role,
            lock_until,
            refresh_token_expiry,
            session_start,
        ) = row

        if lock_until > current_time:
            conn.rollback()
            return None

        if not session_start:
            conn.rollback()
            return None

        absolute_expiry = (
            session_start
            + ABSOLUTE_SESSION_EXPIRE_DAYS * 24 * 60 * 60
        )

        if current_time > absolute_expiry:
            conn.rollback()
            return None

        if refresh_token_expiry < current_time:
            conn.rollback()
            return None

        access_token = create_access_token({
            "sub": emp_id,
            "role": role,
            "session_start": session_start,
        })

        new_refresh_token = secrets.token_urlsafe(32)

        new_refresh_token_hash = hashlib.sha256(
            new_refresh_token.encode()
        ).hexdigest()

        new_expiry = (
            current_time
            + REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60
        )

        # setting new refresh and access token in the database
        # for the particular user
        cur.execute(
            """
            UPDATE users
            SET refresh_token = %s,
                refresh_token_expiry = %s
            WHERE emp_id = %s
              AND refresh_token = %s
            """,
            (
                new_refresh_token_hash,
                new_expiry,
                emp_id,
                refresh_token_hash,
            )
        )

        if cur.rowcount != 1:
            conn.rollback()
            return None

        conn.commit()

        return {
            "access_token": access_token,
            "refresh_token": new_refresh_token,
        }

    except Exception:
        # Mark connection as bad so it is not reused
        bad_connection = True

        # Try to rollback the failed transaction
        try:
            conn.rollback()
        except Exception:
            pass

        # Re-raise the original error
        raise

    finally:
        # Safely close the cursor and return/discard the connection
        if cur:
            cur.close()

        connection_pool.putconn(conn, close=bad_connection)