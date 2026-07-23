# API_REFERENCE.md

# Important

This document describes the API contract for the Role-Based AI Assistant.

Unless an intentional API version change is made, request/response formats should be preserved to maintain compatibility between the frontend and backend.

Base URL (Development)

```
http://localhost:8000
```

All request and response bodies use JSON unless otherwise specified.

Authentication is handled using HttpOnly cookies.

The frontend communicates with the backend using:

```
withCredentials: true
```

---

# Authentication Endpoints

---

## POST /login

Authenticate an existing user.

Authentication Required

❌ No

Request

```json
{
    "emp_id": "emp001",
    "password": "password123"
}
```

Response

```json
{
    "message": "Login successful"
}
```

Side Effects

- Creates Access Token
- Creates Refresh Token
- Stores Refresh Token Hash
- Sets HttpOnly Cookies
- Initializes authenticated session

Possible Errors

```
400 Invalid credentials

403 Account locked

500 Internal server error
```

---

## GET /me

Returns the currently authenticated user.

Authentication Required

✅ Yes

Request

No request body.

Response

```json
{
    "name": "John Doe",
    "emp_id": "emp001",
    "role": "finance",
    "accessible_folders": [
        "finance",
        "general"
    ]
}
```

Purpose

Used by:

- AuthContext
- Dashboard
- Session Verification
- Protected Routes

---

## POST /refresh

Refresh an expired Access Token.

Authentication Required

Refresh Token Cookie

Request

No request body.

Response

```json
{
    "message": "Token refreshed"
}
```

Side Effects

- Generates new Access Token
- Generates new Refresh Token
- Rotates Refresh Token
- Updates HttpOnly Cookies

Automatically called by the Axios response interceptor.

Frontend pages should never call this endpoint directly.

Possible Errors

```
401 Refresh token missing

401 Invalid or expired refresh token

500 Internal server error
```

---

## POST /logout

Logout the current user.

Authentication Required

✅ Yes

Request

No request body.

Response

```json
{
    "message": "Logged out successfully"
}
```

Side Effects

- Deletes Refresh Token Hash
- Resets Session
- Clears Authentication Cookies

---

# Password Endpoints

---

## POST /set-password

Used only for first-time users.

Authentication Required

❌ No

Request

```json
{
    "emp_id": "emp001",
    "password": "newPassword123"
}
```

Response

```json
{
    "message": "Password set successfully"
}
```

Purpose

Initial password setup.

Can only be performed once.

---

## POST /change-password

Changes the current user's password.

Authentication Required

✅ Yes

Request

```json
{
    "current_password": "oldPassword",
    "new_password": "newPassword123"
}
```

Response

```json
{
    "message": "Password changed successfully"
}
```

Side Effects

- Updates Argon2 Password Hash
- Deletes Refresh Token
- Invalidates Existing Sessions
- Forces User Login

---

# AI Endpoint

---

## POST /ask

Submit a question to the AI assistant.

Authentication Required

✅ Yes

Request

```json
{
    "question": "What is the engineering leave policy?"
}
```

Response

```json
{
    "answer": "...",
    "sources": [
        "engineering_policy.pdf",
        "employee_handbook.pdf"
    ]
}
```

Purpose

Main Retrieval-Augmented Generation (RAG) endpoint.

Pipeline

```
Authenticate User

↓

Verify Session

↓

Load Current User

↓

Load Role

↓

Determine Accessible Folders

↓

Semantic Retrieval (MMR)

↓

Prompt Construction

↓

Groq Qwen 3.6 27B

↓

Return Answer
```

Possible Errors

```
401 Unauthorized

403 Forbidden

500 Internal Server Error
```

---

# Dataset Endpoints

---

## GET /dataset

Returns the list of available documents.

Authentication

Depends on endpoint usage.

Behavior

- Home page returns the public dataset.
- Dashboard returns only documents accessible to the authenticated user's role.

Purpose

Used by the Dataset page to display available documents.

---

## GET /document/{document_id}

Returns the contents of a selected document.

Authentication

Depends on endpoint usage.

Behavior

- Public documents are accessible from the Home page.
- Protected documents require authentication and RBAC validation.

Purpose

Used by the Dataset document viewer.

---

# Administration Endpoints

---

## POST /admin/create-user

Creates a new employee.

Authentication Required

✅ Admin

Request

```json
{
    "emp_id": "emp010",
    "name": "Alice",
    "role": "finance"
}
```

Response

```json
{
    "message": "User created successfully"
}
```

Purpose

Administrative employee provisioning.

Validation

- Duplicate Employee ID Detection
- Role Validation
- Admin-only Access

---

# Authentication Strategy

The frontend never stores JWTs.

Authentication relies on:

- HttpOnly Cookies
- Axios withCredentials
- Automatic Token Refresh
- Refresh Token Rotation
- Session Verification

---

# Response Codes

```
200 Success

201 Resource Created

400 Bad Request

401 Unauthorized

403 Forbidden

404 Not Found

500 Internal Server Error
```

---

# Frontend Service Layer

The frontend communicates with the backend exclusively through the services layer.

Available service functions:

```
login()

logout()

getCurrentUser()

refresh()

askQuestion()

setPassword()

changePassword()

createUser()

getDataset()

getDocument()
```

Pages never communicate directly with Axios.

Networking remains centralized inside the services layer.

---

# Notes

- Access Tokens are short-lived JWTs.
- Refresh Tokens are automatically rotated.
- Authorization is enforced before document retrieval.
- Role information is always loaded from PostgreSQL.
- Authentication uses HttpOnly cookies exclusively.
- Axios automatically retries failed requests after successful token refresh.
- Authentication and authorization remain separate concerns.