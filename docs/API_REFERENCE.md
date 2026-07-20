# API_REFERENCE.md

# Important: 

This document describes the API contract. Maintain request/response formats unless there is an intentional API version change. Frontend and backend are designed around these endpoint contracts.

# API Reference

This document describes every backend endpoint exposed by the Role-Based AI Assistant.

Base URL (Development)

```
http://localhost:8000
```

All request and response bodies use JSON unless otherwise specified.

Authentication is handled using HttpOnly cookies.

The frontend communicates using:

```
withCredentials: true
```

---

# Authentication Endpoints

---

## POST /login

Authenticate an existing user.

Authentication Required:

❌ No

Request

```json
{
    "emp_id": "emp001",
    "new_password": "password123"
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

Possible Errors

```
400 Invalid credentials

403 Account locked

500 Internal server error
```

---

## GET /me

Returns the currently authenticated user.

Authentication Required:

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
- Session verification

---

## POST /refresh

Refresh an expired Access Token.

Authentication Required:

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
- Updates cookies

Called automatically by the Axios interceptor.

Frontend should never call this directly under normal operation.

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
- Resets session_start
- Clears cookies

---

# Password Endpoints

---

## POST /set-password

Used only by first-time users.

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

Can only be used once.

---

## POST /change-password

Changes the current password.

Authentication Required

✅ Yes

Request

```json
{
    "current_password": "oldPassword",
    "new_password": "newPassword"
}
```

Response

```json
{
    "message": "Password changed successfully"
}
```

Side Effects

- Updates Argon2 password hash
- Deletes Refresh Token
- Invalidates all sessions
- Requires user to login again

---

# RAG Endpoint

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
    "answer": "Employees are allowed ...",
    "sources": [
        "engineering_policy.pdf",
        "employee_handbook.pdf"
    ]
}
```

Purpose

Main RAG endpoint.

Pipeline:

```
Authenticate

↓

Verify Session

↓

Load Role

↓

Determine Accessible Folders

↓

MMR Retrieval

↓

Prompt Construction

↓

Groq LLaMA

↓

Return Answer
```

Possible Errors

```
401 Unauthorized

403 Account Locked

500 Internal Error
```

---

# Admin Endpoints

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

Administrative user provisioning.

---

# Authentication Strategy

The frontend never stores JWTs.

Authentication relies on:

- HttpOnly Cookies
- Axios withCredentials
- Automatic refresh interceptor

---

# Response Codes

```
200 Success

201 Resource Created

400 Invalid Request

401 Unauthorized

403 Forbidden

404 Not Found

500 Internal Server Error
```

---

# Frontend Usage

The frontend communicates only through:

```
services/auth.js
```

Available functions:

```
login()

logout()

getCurrentUser()

refresh()

askQuestion()
```

Future additions:

```
setPassword()

changePassword()

createUser()
```

Pages never communicate with Axios directly.

Networking remains centralized inside the services layer.

---

# Notes

- Access Tokens are short-lived JWTs.
- Refresh Tokens are automatically rotated.
- Authorization is enforced before document retrieval.
- Role information is always loaded from PostgreSQL.
- HttpOnly cookies are required for every authenticated request.
- Axios automatically retries requests after successful token refresh.