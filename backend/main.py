from fastapi import FastAPI, HTTPException   # FastAPI framework, HTTPException for error responses
from pydantic import BaseModel               # Used to define request body structure (JSON input)

from rag_engine import ask_question          # Your existing RAG function
from auth import authenticate_user, create_access_token, verify_token   # NEW: Import auth functions
from auth import create_refresh_token,load_users,save_users 
import time  #used for session tracking and lock checks

app = FastAPI()                              # Create FastAPI app


# NEW: Model for login request body
class LoginRequest(BaseModel):
    username: str
    password: str


# CHANGED: Old QueryRequest had role + question
# Now role is removed and replaced with token
class QueryRequest(BaseModel):
    token: str
    question: str


# NEW: Login endpoint  (When the user hits login)
@app.post("/login")
def login(request: LoginRequest):

    # NEW: Check if username + password are correct, we go to auth.py file to check
    user = authenticate_user(request.username, request.password)
    
    if user=="LOCKED": #if account is locked
        raise HTTPException(status_code=403, detail="Account locked. Try again later.") #We send hhtp response back to the client and the client understands the type of error
    
    if user=="LAST_ATTEMPT":   #if account has last attempt remaining
        raise HTTPException(status_code=401, detail="Invalid username or password. Warning. Last Attempt left.")

    # If authentication fails → return error, user has entered either wrong username or password
    if not user:   
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    #The session clock starts, absolute session is calculated with env variable
    users=load_users() #We access the users.json file
    users[request.username.lower()]["session_start"] = time.time() #We first ask for the details of the specific user and then it overwrites the prev session time everytime 
                                                                   #when we login and starts the session time from fresh login
    save_users(users)  #write updated dictionary back to users.json

    # NEW: Create JWT token with username and role inside payload
    access_token = create_access_token({   #We call the auth.py file and it creates access token and returns it to main
        "sub": request.username.lower(),
        "role": user["role"]              
    })
    #we are passing username and his role as parameters to auth file, which creates jwt token and returns it to main in access_token variable
    
    refresh_token = create_refresh_token(request.username.lower()) #We call creeate_refresh function and  pass the username as parameter to create_refresh function defined in auth file, which creates and returns the refresh token to us

    # Send token back to user, “We send this data as an HTTP response back to the client”
    return { 
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"  #Bearer = whoever holds the token is the user                                          It tells the client how to use the token, typically indicating it should be sent as a Bearer token in the Authorization header.
           }

class RefreshRequest(BaseModel):
    refresh_token: str
    
@app.post("/refresh")   # called when frontend requests token refresh 
def refresh(request: RefreshRequest):      #defining refresh api
    from auth import refresh_access_token
    tokens = refresh_access_token(request.refresh_token)

    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

    return tokens  #shows both tokens to the user as http response, visible on frontend
    
'''# Client sends refresh token (string) to this endpoint
# This endpoint is called by frontend when access token expires (not a UI button defined here)

# We call refresh_access_token() from auth.py and pass the refresh token
# Control goes to auth.py where:
#   - The refresh token is validated (hash match, expiry, session, lock checks)
#   - If valid, a NEW access token and NEW refresh token are generated
#   - The new refresh token replaces the old one in users.json (rotation)

# The function returns the new tokens back to main.py

# If validation fails → raise HTTPException (401 Unauthorized)
# If successful → return new tokens as HTTP response (JSON) to the client(user)'''
    
@app.post("/ask")  #we click the ask button
def ask_ai(request: QueryRequest):   #defining the ask endpoint here
    
    # NEW: Verify token and extract payload from JWT, verify function is called here and executes in auth.py file
    payload = verify_token(request.token) #We pass access token as argument, the function returns payload which contains username, role and the expiry date of the token

    # If token invalid or expired → error is shown to user
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    username = payload["sub"]  #We store the username of user from the returned payload in the username variable,

    # We Load users from database
    users = load_users()       #this function executes in auth.py and it returns the entire users.json file and we store that in users variable as a dictionary
    user = users.get(username)  # we use username from payload to fetch user from database, so that we can also fetch the user's role from database 

    if not user:  #if user not found, error is raised
        raise HTTPException(status_code=404, detail="User not found")

    # Check if account is locked, if it is, then error is raised
    #If the account is locked, then even if access token(JWT) is valid, you can't ask and error is raised
    if user["lock_until"] > time.time():
        raise HTTPException(status_code=403, detail="Account is locked")

    # IMPORTANT: # Get role from database (do NOT trust role from JWT for security reasons), also if admin changed user's role, so we are checking to make sure the role is correct 
    role = user["role"]

    # Pass user's role and the question
    # This function performs RBAC (Role Based Authentication Checks), retrieves documents, calls LLM, and returns answer
    result = ask_question(role, request.question) 
    
    return result  #Return the result (answer + sources) as HTTP response to the client
    