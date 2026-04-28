import json #used for reading and writing json files, used in load_users(), save_users() function
import os #used for reading environment variables
from datetime import datetime, timedelta, timezone  #used for creating expiry times for jwt
import time #The time module is used to get the current Unix timestamp (seconds since 1970 UTC), which is used for session tracking, account lock timing, token expiry, and security checks

from dotenv import load_dotenv #loads environment file into the environment
from jose import jwt  #used for creating and verifying jwt
from passlib.context import CryptContext #used for hashing passwords (Argon2 algorithm)

import secrets #generates secure tokens(used for refresh tokens)
import hashlib #used for hashing refresh token before storing

# This reads our .env file and store values into  environment
load_dotenv()

#This block loads configuration values from the .env file and stores them as variables used for JWT, token expiry, and security settings
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

#These variables  are stored as int and float,cuz they are stored as strings in env file, and we will need these values to do some math
ACCESS_TOKEN_EXPIRE_MINUTES = float(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))

REFRESH_TOKEN_EXPIRE_DAYS = float(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS"))
ABSOLUTE_SESSION_EXPIRE_DAYS = float(os.getenv("ABSOLUTE_SESSION_EXPIRE_DAYS"))

MAX_FAILED_ATTEMPTS = int(os.getenv("MAX_FAILED_ATTEMPTS"))  
LOCKOUT_BASE_MINUTES = float(os.getenv("LOCKOUT_BASE_MINUTES"))
LOCKOUT_RESET_HOURS = float(os.getenv("LOCKOUT_RESET_HOURS"))

# Use Argon2 for hashing
#It creates a password hashing manager
#“This object knows how to hash passwords and verify them using Argon2
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


# Load users from users.json
'''# Load all users from users.json file
# Called from multiple endpoints (/login, /ask) and auth functions
# Returns data as a dictionary (username → user details)'''

def load_users():
    with open("users.json", "r") as file:
        return json.load(file)

'''
# Save updated users data to users.json file
# Called after modifying user data (login, failed attempts, tokens, etc.)
'''
def save_users(users):
    with open("users.json", "w") as file:
        json.dump(users, file, indent=4)


# Verify password
'''
# Verify user password using Argon2 hashing
# Called in this file only inside authenticate_user() during login
# Compares plain password with stored hash
'''

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Authenticate user during login
#called from login endpoint
def authenticate_user(username, password):
    #loads users
    users = load_users()
    username = username.lower()
    user = users.get(username)

    if not user:
        return None

    current_time = time.time()  #this gets current time

    # Auto reset after 24 hours
    if user["last_failed_login"] != 0:
        if current_time - user["last_failed_login"] > (LOCKOUT_RESET_HOURS * 3600):
            user["failed_attempts"] = 0
            user["lock_count"] = 0

    # Check if account is locked
    if user["lock_until"] > current_time:
        return "LOCKED"

    # Check password
    if verify_password(password, user["password_hash"]):
        # Correct password → reset everything
        user["failed_attempts"] = 0
        user["lock_count"] = 0
        user["lock_until"] = 0
        user["last_failed_login"] = 0

        save_users(users)
        return user           #returns user object to main, basically it returns the specific user's data as dictionary

    else:
        # Wrong password
        user["failed_attempts"] += 1
        user["last_failed_login"] = current_time

        # If 4 attempts → warning
        if user["failed_attempts"] == MAX_FAILED_ATTEMPTS-1:
            save_users(users)
            return "LAST_ATTEMPT"

        # Lock account after 5 failed attempts
        if user["failed_attempts"] >= MAX_FAILED_ATTEMPTS:
            user["lock_count"] += 1
            lock_minutes = LOCKOUT_BASE_MINUTES * (2 ** (user["lock_count"] - 1))
            user["lock_until"] = current_time + (lock_minutes * 60)
            user["failed_attempts"] = 0

            save_users(users)
            return "LOCKED"

        save_users(users)
        return None             #go back to main, login fails

# Create JWT token, called from login endpoint
def create_access_token(data: dict):       #username and role is sent as parameter here
    to_encode = data.copy()             

    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)  #creates expiry time
    to_encode.update({"exp": int(expire.timestamp())})  # convert to UNIX timestamp, add expiry to payload of jwt

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)  #encodes the jwt, signs it using secret key
    return encoded_jwt    #returns the jwt payload to main

#Creating Refresh Token, called from login endpoint
def create_refresh_token(username: str):
    users = load_users()

    refresh_token = secrets.token_urlsafe(32)  #generates random token string
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()  #hashes the token using this algo
    
    current_time=time.time()  #gets current time
    
    refresh_token_expiry= current_time + (REFRESH_TOKEN_EXPIRE_DAYS *24 *60 *60)  #creates refresh expiry time
    
    users[username]["refresh_token"] = refresh_token_hash            #hash and its expiry time is stored in db
    users[username]["refresh_token_expiry"] = refresh_token_expiry

    save_users(users)  #saves the db

    return refresh_token    #returns the refresh token to main

#Function to refresh the access token, called from refresh end point
def refresh_access_token(refresh_token: str): #recieves the prev refresh token as parameter
    users = load_users()     #loads the full user database
    refresh_token_hash = hashlib.sha256(refresh_token.encode()).hexdigest() #hashes the incoming token
    current_time = time.time() #gets current time

    for username, user in users.items(): #loop through users to see which user holds the refresh token

        # Check if refresh token matches
        if user.get("refresh_token") == refresh_token_hash:
            
            #if account is locked, then deny refresh
            if user["lock_until"] > current_time:
                return None
            
            #Absolute session expiry check
            session_start= user.get("session_start")
            #If no active session, i.e, session=0 or missing in users.json, refresh not allowed(basically if user not logged in properly), can happen if you have the refresh token but the absolute time expires
            if not session_start:
                return None
            
            absolute_expiry=session_start+ (ABSOLUTE_SESSION_EXPIRE_DAYS * 24 * 60 *60) #calculate absolute expiry time
            
            if current_time>absolute_expiry:  #if absolute expiry time has passed, i.e, the refresh token becomes invalid, refresh not allowed 
                return None

            # Check if refresh token expired
            if user.get("refresh_token_expiry", 0) < current_time:
                return None

            # Create new access token, if refresh token matches
            access_token = create_access_token({    #calls the create_access_token in this file only
                "sub": username,
                "role": user["role"]
            })

            # Create NEW refresh token (rotation), as refresh token has matched
            new_refresh_token = secrets.token_urlsafe(32) #new token generated
            new_refresh_token_hash = hashlib.sha256(new_refresh_token.encode()).hexdigest()  #we hash the token using this algorithm
            new_expiry = current_time + (REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 *60)  #we set the expiry time for the token

            user["refresh_token"] = new_refresh_token_hash  #old token and old expiry time is replaced by new refresh token and new expiry time
            user["refresh_token_expiry"] = new_expiry

            save_users(users) #We save the changes to the database

            return {                                #goes back to main, returns both the tokens
                "access_token": access_token,
                "refresh_token": new_refresh_token
            } 

    return None       #if refresh token doesn't match , return none to main file

# Verify JWT token
from jose import JWTError, ExpiredSignatureError    #These are exceptions thrown by jwt.decode(), and without catching them your program will crash.

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM]) #internally the function checks signature, algorithm, expiry and the token format
        return payload       #if everything is valid, jwt token string is returned to main
    
    except ExpiredSignatureError:
        print("Token expired")         #if token has expired
        return None
    
    except JWTError:
        print("Invalid token")           #if token is tampered, malformed, wrong signature
        return None