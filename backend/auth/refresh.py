import hashlib
import secrets
import time

from backend.shared_cons import connection_pool

from .database import update_user, get_user
from .jwt_handler import create_access_token
from .config import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    ABSOLUTE_SESSION_EXPIRE_DAYS,
)




#Creating Refresh Token, called from login endpoint
def create_refresh_token(emp_id: str):
    emp_id = emp_id.lower()
    user = get_user(emp_id)
    if not user:
        return None

    refresh_token = secrets.token_urlsafe(32)  #generates random token string
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()  #hashes the token using this algo
    
    current_time = time.time()  #gets current time
    
    refresh_token_expiry = current_time + (REFRESH_TOKEN_EXPIRE_DAYS *24 *60 *60)  #creates refresh expiry time
    
    user["refresh_token"] = refresh_token_hash            #hash and its expiry time is stored in db
    user["refresh_token_expiry"] = refresh_token_expiry

    update_user(emp_id, user)  #saves the db

    return refresh_token    #returns the refresh token to main





#Function to refresh the access token, called from refresh end point

def refresh_access_token(refresh_token: str): #recieves the prev refresh token as parameter
    

    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest() #hashes the incoming token
    current_time = time.time() #gets current time

    # Connect to DB
    conn = connection_pool.getconn()
    try:
        cur = conn.cursor()

        # Fetch user directly using refresh token
        cur.execute("SELECT * FROM users WHERE refresh_token = %s", (refresh_token_hash,))
        row = cur.fetchone()

        cur.close()
    finally:
        connection_pool.putconn(conn)

    if not row:
        return None

    emp_id = row[0]

    user = {
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

    #if account is locked, then deny refresh
    if user["lock_until"] > current_time:
        return None
    
    #Absolute session expiry check
    session_start = user.get("session_start")

    if not session_start:
        return None
    
    absolute_expiry = session_start + (ABSOLUTE_SESSION_EXPIRE_DAYS * 24 * 60 *60)
    
    if current_time > absolute_expiry:
        return None

    # Check if refresh token expired
    if user.get("refresh_token_expiry", 0) < current_time:
        return None

    # Create new access token
    access_token = create_access_token({
        "sub": emp_id,
        "role": user["role"],
        "session_start": user["session_start"]
    })

    # Create NEW refresh token (rotation)
    new_refresh_token = secrets.token_urlsafe(32)
    new_refresh_token_hash = hashlib.sha256(new_refresh_token.encode()).hexdigest()
    new_expiry = current_time + (REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 *60)

    user["refresh_token"] = new_refresh_token_hash
    user["refresh_token_expiry"] = new_expiry

    update_user(emp_id, user)  # replace save_users

    return {
        "access_token": access_token,
        "refresh_token": new_refresh_token
    }
    