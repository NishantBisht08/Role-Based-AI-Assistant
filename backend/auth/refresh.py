import hashlib
import secrets
import time

from backend.shared_cons import connection_pool

from .database import update_user, get_user, db_rotate_refresh_token
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

def refresh_access_token(refresh_token: str):
    
    #hashing the old refresh token
    refresh_token_hash = hashlib.sha256(
        refresh_token.encode()
    ).hexdigest()                                       #verifies whether existing refresh token is valid or not

    return db_rotate_refresh_token(refresh_token_hash) 
    