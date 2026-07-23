from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Depends
from backend.rag_engine.rbac import ROLE_FOLDERS

# Document service for listing dataset documents
from backend.document_service.service import (
    list_all_documents,
    list_role_documents,
    get_document,
    get_public_document,
)

from fastapi import FastAPI, HTTPException, Request   # FastAPI framework, HTTPException for error responses
from pydantic import BaseModel, Field               # Used to define request body structure (JSON input)

import time  #used for session tracking and lock checks

#from backend.rag_engine import ask_question          # Your existing RAG function  
from backend.shared_cons import connection_pool

from backend.auth import (           #importing all the functions defined in the auth folder
    authenticate_user,
    create_access_token,
    verify_token,
    create_refresh_token,
    refresh_access_token,
    logout_user,
    get_user,
    update_user,
    set_user_password,
    change_user_password,
    get_current_user,
)

from backend.auth.config import ABSOLUTE_SESSION_EXPIRE_DAYS, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS

from backend.auth.config import CLIENT_URL

app = FastAPI()                              # Create FastAPI app

app.add_middleware(
    CORSMiddleware,

    allow_origins=[CLIENT_URL],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

ALLOWED_ROLES = {"admin", "hr", "engineering", "employee", "marketing", "finance", "c-level"} #List for allowed folders




# NEW: Model for login request body
class LoginRequest(BaseModel):
    emp_id: str
    password: str


# CHANGED: Old QueryRequest had role + question
# Now role is removed and replaced with token
class QueryRequest(BaseModel):
    token: str
    question: str
    
    

# ── Public Dataset Endpoint ───────────────────────────────────────────────────
# Returns metadata for every document in the dataset.
# This endpoint is public and is used by the Home page Dataset Viewer.
@app.get("/dataset")
def get_dataset():

    return {
        "documents": list_all_documents()
    }
    

# ── Role-Based Dataset Endpoint ───────────────────────────────────────────────
# Returns only the documents the authenticated user is allowed to access.
# This endpoint is used by the Dashboard Dataset Viewer.

@app.get("/documents")
def get_documents(current_user: dict = Depends(get_current_user)):
    
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )    

    return {
             "documents": list_role_documents(current_user["role"])
           }
    

# ── Get Document Endpoint ─────────────────────────────────────────────────────
# Returns the contents of a requested document after verifying RBAC permissions.
# Used by the frontend Dataset Viewer to fetch and display document contents.

@app.get("/documents/{document_id}")
def get_document_content(document_id: str, current_user: dict = Depends(get_current_user) ):
    
    if not current_user:
            raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )

    document = get_document(
        document_id,
        current_user["role"]
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found or access denied."
        )

    return document



# Returns the contents of a single document from the public dataset.
# This endpoint is intentionally public so visitors can explore
# every document from the Home page.
@app.get("/dataset/{document_id}")

def public_document(document_id: str):

    # Retrieve the requested document.
    document = get_public_document(document_id)

    # Return 404 if the document doesn't exist.
    if document is None:

        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    return document



# NEW: Login endpoint  (When the user hits login)
@app.post("/login")
def login(request: LoginRequest):

    # NEW: Check if emp_id + password are correct, we go to auth.py file to check
    user = authenticate_user(request.emp_id, request.password)
    
    if user=="LOCKED": #if account is locked
        raise HTTPException(status_code=403, detail="Account locked. Try again later.") #We send hhtp response back to the client and the client understands the type of error
    
    if user=="LAST_ATTEMPT":   #if account has last attempt remaining
        raise HTTPException(status_code=401, detail="Invalid emp_id or password. Warning. Last Attempt left.")

    # If authentication fails → return error, user has entered either wrong emp_id or password
    if not user:   
        raise HTTPException(status_code=401, detail="Invalid emp_id or password")
    
    #lowering emp_id so that emp101 and EMP101 is same
    emp_id = request.emp_id.lower()

    user_db = get_user(emp_id)   # fetch user from DB by his emp_id
    
    if not user_db:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_db["session_start"] = time.time()   #The session clock starts, it is updated with current time, everytime we login, session time resets
    update_user(emp_id, user_db)   # save updated data to DB

    session_start = user_db["session_start"]   #session_start variable holds the session start time which we will include in our jwt
     
    # NEW: Create JWT token with emp_id and role inside payload
    access_token = create_access_token({   #We call the auth.py file and it creates access token and returns it to main
        "sub": emp_id,
        "role": user["role"] ,     
        "session_start": session_start     #adding session_start_time in jwt   
    })
    #we are passing emp_id and his role as parameters to auth file, which creates jwt token and returns it to main in access_token variable
    
    refresh_token = create_refresh_token(emp_id) #We call creeate_refresh function and  pass the emp_id as parameter to create_refresh function defined in auth file, which creates and returns the refresh token to us

    # Send token back to user, "We send this data as an HTTP response back to the client"
    
    response = JSONResponse(
           content={
                      "message": "Login successful"
                   }
               )

    response.set_cookie(
                    key="access_token",
                    value=access_token,
                    httponly=True,
                    secure=True,          # Change to True after HTTPS deployment
                    samesite="none",
                    max_age=int(ACCESS_TOKEN_EXPIRE_MINUTES * 60)
                )

    response.set_cookie(
                    key="refresh_token",
                    value=refresh_token,
                    httponly=True,
                    secure=True,          # Change to True after HTTPS deployment
                    samesite="none",
                    max_age=int(REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60)
                   )

    return response

# Returns information about the currently logged-in user
@app.get("/me")
def get_me(request: Request):

    # Authenticate the user using the HttpOnly cookie
    user = get_current_user(request)

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )

    return {
        "emp_id": user["emp_id"],
        "name": user["name"],
        "role": user["role"],
        "accessible_folders": ROLE_FOLDERS.get(user["role"], []) #if folder is empty, return nothing
    }
    
    


    
@app.post("/refresh")   # called when frontend requests token refresh 
def refresh(request: Request):      #defining refresh api
    
    refresh_token = request.cookies.get("refresh_token")
    
    if not refresh_token:
        raise HTTPException(
        status_code=401,
        detail="Refresh token missing"
    )
    
    tokens = refresh_access_token(refresh_token)

    if not tokens:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    
    response = JSONResponse(
           content={
                      "message": "Token refreshed successfully"
                   }
               )

    response.set_cookie(
                    key="access_token",
                    value=tokens["access_token"],
                    httponly=True,
                    secure=True,          # Change to True after HTTPS deployment
                    samesite="none",
                    max_age=int(ACCESS_TOKEN_EXPIRE_MINUTES * 60)
                )

    response.set_cookie(
                    key="refresh_token",
                    value=tokens["refresh_token"],
                    httponly=True,
                    secure=True,          # Change to True after HTTPS deployment
                    samesite="none",
                    max_age=int(REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60)
                   )

    return response  # Sends the new HttpOnly cookies back to the browser
    
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


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000
    )
    
'''    
@app.post("/ask")  #we click the ask button
def ask_ai(request: Request, query: QueryRequest):   #defining the ask endpoint here, user provides token and question
    
    access_token = request.cookies.get("access_token")  #reading JWT token from browser 

    if not access_token:
              raise HTTPException(
              status_code=401,
              detail="Access token missing"
            )
    
    # NEW: Verify token and extract payload from JWT, verify function is called here and executes in auth.py file
    payload = verify_token(access_token) #We pass access token as argument, the function returns payload which contains emp_id, role and the expiry date of the token

    # If token invalid or expired → error is shown to user
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    emp_id = payload["sub"].lower()  #We store the emp_id of user from the returned payload in the emp_id variable,

    # We Load users from database
   
    user = get_user(emp_id)   # fetch user from DB using his emp_id

    if not user:  #if user not found, error is raised
        raise HTTPException(status_code=404, detail="User not found")
    
    # absolute session expiry check
    ABSOLUTE_SESSION_EXPIRE_SECONDS = ABSOLUTE_SESSION_EXPIRE_DAYS * 86400
    if time.time() > user.get("session_start", 0) + ABSOLUTE_SESSION_EXPIRE_SECONDS:
            raise HTTPException(status_code=401, detail="Session expired. Please login again.")
    
    token_session = payload.get("session_start")

    #if user isn't logged in (he has logged out), session_start_time wouldn't match. Here, we are checking session_start time of jwt with Database
    if token_session != user.get("session_start"):
        raise HTTPException(status_code=401, detail="Session expired. Please login again.")

    # Check if account is locked, if it is, then error is raised
    #If the account is locked, then even if access token(JWT) is valid, you can't ask and error is raised
    if user["lock_until"] > time.time():
        raise HTTPException(status_code=403, detail="Account is locked")

    # IMPORTANT: # Get role from database (do NOT trust role from JWT for security reasons), also if admin changed user's role, so we are checking to make sure the role is correct 
    role = user["role"]
    
    question = query.question.strip()
    
    if not question:
        raise HTTPException(
        status_code=400,
        detail="Question cannot be empty."
    )

    # Pass user's role and the question
    # This function performs RBAC (Role Based Authentication Checks), retrieves documents, calls LLM, and returns answer
    result = ask_question(role, question)
    
    return result  #Return the result (answer + sources) as HTTP response to the client '''
    
    
    
#When logout endpoint is called
@app.post("/logout")
def logout(request: Request):
    
    refresh_token = request.cookies.get("refresh_token")
    
    if not refresh_token:
        raise HTTPException(
        status_code=401,
        detail="Refresh token missing"
    )

    #we call the logout function and it executes in auth.py
    #on successful logout, true is returned and on unsuccessful logout, false is returned, and we store it in result
    result = logout_user(refresh_token)   

    #if logout is unsuccessful
    if not result: 
       raise HTTPException(status_code=401, detail="Invalid refresh token")

    #if logout is successful
    response = JSONResponse(
            content={
                     "message": "Logged out successfully"
                    }
                )
    
    response.delete_cookie(
             key="access_token"
         )
     
    response.delete_cookie(
             key="refresh_token"
         )
    
    return response 
    


#End points for admin
class CreateUserRequest(BaseModel):
    emp_id: str
    role:   str
    name:   str
    
# if admin wants to add a new user
@app.post("/admin/create-user")
def create_user(request: CreateUserRequest, 
                current_user = Depends(get_current_user)
                ):

    if not current_user:
        raise HTTPException(
        status_code=401,
        detail="Not authenticated"
    )
        
    if current_user["role"] != "admin":
        raise HTTPException(
        status_code=403,
        detail="Not authorized"
    )

    emp_id = request.emp_id.strip().lower()  #converting new emp_id provided by admin of the new user to lowercase
    
    if not emp_id:
        raise HTTPException(
        status_code=400,
        detail="Employee ID cannot be empty."
    )
    
    #Removes spaces from start and end, ex: "  Sid " becomes "Sid" and "  " becomes ""
    name = request.name.strip()
    #if admin has enetered an empty name, then we raise an error
    if not name:
        raise HTTPException(status_code=400, detail="Name cannot be empty")

    # Check if newly created user's emp_id already exists in db
    existing_user = get_user(emp_id)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    #getting the new user's role from admin
    role = request.role.strip().lower()

    #checking if admin created user role exists or not 
    if role not in ALLOWED_ROLES:
            raise HTTPException(status_code=400, detail="Invalid role")
    
    if role == "admin":
        raise HTTPException(status_code=403, detail="Cannot create admin users")

    # Create new user WITHOUT password
    conn = connection_pool.getconn()
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
        cur.close()
    finally:
        connection_pool.putconn(conn)

    return {"message": f"User {emp_id} created successfully"}



#allowing user to set password only if user exists and password is currently empty
class SetPasswordRequest(BaseModel):
    emp_id: str
    new_password: str
    
#user enters his emp_id and new_password to set
@app.post("/set-password")
def set_password(request: SetPasswordRequest):
    
    #Empty password (like "" or "  ") is not allowed
    new_password = request.new_password.strip()
    if not new_password:
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    emp_id = request.emp_id.lower()

    user = get_user(emp_id)

    #if user does not exist, we throw error
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # CASE 1: First-time password setup (no password exists)
    # we allow this WITHOUT requiring token (onboarding flow)
    if not user.get("password_hash"):
        result = set_user_password(emp_id,new_password)

        if result == "USER_NOT_FOUND":
            raise HTTPException(status_code=404, detail="User not found")

        return {"message": "Password set successfully"}

    # CASE 2: Password already exists → require authentication
    
    raise HTTPException(status_code=400, detail="Password already exists. Use change/password instead")   


#endpoint for changing password
class ChangePasswordRequest(BaseModel):
    emp_id: str
    old_password: str
    new_password: str
    
#user enters his emp_id, old password to verify and the new password he wants to replace the old password with
@app.post("/change-password")
def change_password(request: ChangePasswordRequest, current_user = Depends(get_current_user)):
    
    if not current_user:
        raise HTTPException(
           status_code=401,
           detail="Not authenticated"
    )

    if request.emp_id.strip().lower() != current_user["emp_id"].lower():
        raise HTTPException(
           status_code=403,
           detail="You can only change your own password."
    )
    
    #old password can't be replaced with Empty password 
    new_password = request.new_password.strip()
    if not new_password:
        raise HTTPException(status_code=400, detail="Password cannot be empty")

    #if password is correct, success message is returned, else some kind of error message is returned
    result = change_user_password(
        current_user["emp_id"],
        request.old_password,
        new_password
    )

    if result == "USER_NOT_FOUND":
        raise HTTPException(status_code=404, detail="User not found")

    if result == "LOCKED":
        raise HTTPException(status_code=403, detail="Account is locked")

    if result == "PASSWORD_NOT_SET":
        raise HTTPException(status_code=400, detail="Password not set yet")

    if result == "WRONG_PASSWORD":
        raise HTTPException(status_code=401, detail="Incorrect old password")

    return {"message": "Password changed successfully. Please login again."}