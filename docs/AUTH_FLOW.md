# AUTH_FLOW.md

# Authentication Flow

This document describes the complete authentication and session management system used by the Role-Based AI Assistant.

---

# Overview

The project uses a secure cookie-based authentication system built around:

- JWT Access Tokens
- Refresh Tokens
- HttpOnly Cookies
- Argon2 Password Hashing
- Refresh Token Rotation
- Absolute Session Expiry
- Role-Based Authorization
- PostgreSQL Session Storage

Authentication and authorization are intentionally separated.

Authentication determines **who the user is**.

Authorization determines **what the user is allowed to access**.

---

# Authentication Architecture

```
React Frontend
        │
        ▼
Axios
(withCredentials)
        │
        ▼
FastAPI
        │
        ▼
Authentication
        │
        ▼
PostgreSQL
```

---

# Authentication Lifecycle

```
Login

↓

Access Token Created

↓

Refresh Token Created

↓

HttpOnly Cookies Returned

↓

Authenticated Requests

↓

Access Token Expires

↓

Automatic Refresh

↓

New Access Token

↓

Continue Session
```

Eventually:

```
Refresh Token Expires

↓

Login Required
```

or

```
Absolute Session Expires

↓

Login Required
```

---

# Login Flow

```
User

↓

POST /login

↓

authenticate_user()

↓

verify_password()

↓

create_access_token()

↓

create_refresh_token()

↓

Store Refresh Token Hash

↓

Set HttpOnly Cookies

↓

Return Success
```

The frontend never stores tokens manually.

The browser automatically stores the cookies.

---

# Access Token

Purpose:

Authenticate every protected request.

Characteristics:

- JWT
- Short lifetime
- Stored as HttpOnly Cookie
- Automatically sent by browser

Contains:

- Employee ID (sub)
- Session Start Timestamp
- Expiration Time

The role is intentionally NOT trusted from the JWT.

---

# Refresh Token

Purpose:

Generate new Access Tokens without forcing the user to login again.

Characteristics:

- Long lifetime
- Stored as HttpOnly Cookie
- Rotated every refresh
- Stored in database only as SHA-256 hash

The plaintext refresh token is never stored inside PostgreSQL.

---

# Automatic Refresh Flow

Frontend uses an Axios response interceptor.

Flow:

```
Protected Request

↓

401 Unauthorized

↓

POST /refresh

↓

Verify Refresh Token

↓

Generate New Tokens

↓

Update Cookies

↓

Retry Original Request
```

Only one refresh request may execute simultaneously.

Concurrent failed requests share the same refresh operation.

---

# Refresh Token Rotation

Every refresh request invalidates the previous refresh token.

Flow:

```
Refresh Token A

↓

Verified

↓

Refresh Token B Created

↓

Database Updated

↓

Token A Invalid

↓

Future Requests Use Token B
```

This prevents refresh token replay attacks.

---

# Session Binding

Every JWT contains a session_start timestamp.

During authentication:

```
JWT session_start

↓

Database session_start

↓

Match ?

↓

YES → Continue

NO → Reject Token
```

This allows logout and password changes to immediately invalidate all previous JWTs.

---

# Logout Flow

```
Logout Request

↓

Delete Refresh Token Hash

↓

Reset session_start

↓

Clear Cookies

↓

Return Success
```

Every previously issued access token immediately becomes invalid because its session_start no longer matches the database.

---

# Password Setup

Used for first-time users.

Flow:

```
Employee ID

↓

Temporary Password Validation

↓

Argon2 Hash

↓

Store Password Hash

↓

Mark Password Initialized

↓

Success
```

This endpoint does not require an existing authenticated session.

---

# Password Change

Flow:

```
Authenticated User

↓

Verify Current Password

↓

Hash New Password

↓

Update Password

↓

Invalidate Refresh Token

↓

Reset Session

↓

Force Login
```

Changing a password immediately logs out all existing sessions.

---

# Account Locking

Purpose:

Protect against brute-force attacks.

Configuration:

- Maximum failed attempts
- Exponential lock duration
- Automatic reset after inactivity

Example:

```
Wrong Password

↓

Attempt 1

↓

Attempt 2

↓

Attempt 3

↓

Attempt 4

↓

Attempt 5

↓

Account Locked
```

Subsequent lock durations increase exponentially.

---

# Absolute Session Expiry

Refresh token rotation does NOT allow infinite sessions.

Each login creates a fixed session lifetime.

```
Login

↓

Session Starts

↓

30 Minutes

↓

Session Ends

↓

Login Required
```

This limits long-term exposure even if refresh tokens continue rotating.

---

# Authorization Flow (RBAC)

Authentication completes before authorization begins.

```
Authenticated User

↓

Load Current User

↓

Read Role From Database

↓

Determine Accessible Folders

↓

Load Correct ChromaDB

↓

Retrieve Documents
```

The role is always loaded from PostgreSQL.

Client-supplied roles are ignored.

JWT roles are not trusted.

---

# Cookie Strategy

Cookies are configured as:

- HttpOnly
- SameSite
- Secure (Production)

Frontend uses:

```
withCredentials: true
```

The frontend never accesses:

- document.cookie
- localStorage
- sessionStorage

Authentication is completely browser-managed.

---

# Security Features

Implemented protections:

- Argon2 password hashing
- JWT expiration
- Refresh token rotation
- Refresh token hashing
- HttpOnly cookies
- Session binding
- Absolute session expiry
- Account lockout
- Exponential lock duration
- Role verification from database
- Logout invalidation
- Password change invalidation

---

# Authentication Modules

```
auth/

config.py
database.py
login.py
logout.py
refresh.py
password.py
jwt_handler.py
auth_guard.py
```

Each module has a single responsibility.

Business logic is intentionally separated to improve maintainability.

---

# Design Principles

The authentication system was designed around the following principles:

- Authentication and authorization remain independent.
- Short-lived access tokens reduce risk.
- Refresh tokens never remain static.
- The backend owns authentication state.
- The frontend never stores authentication tokens.
- Role information is always loaded from the database.
- Every security-sensitive operation invalidates old sessions.
- Business logic remains modular rather than centralized in one file.

---

# Summary

Authentication provides:

- Secure login
- Secure logout
- Automatic session refresh
- Enterprise session management
- Brute-force protection
- Role-aware authorization
- Stateless JWT authentication combined with server-side session validation

This design intentionally balances performance (JWT authentication) with security (server-side session validation and refresh token storage).


# Important Implementation Notes

- The frontend uses HttpOnly cookies exclusively.
- Never suggest storing JWTs in localStorage.
- Role information must always come from the database, not the JWT.
- Refresh token rotation is already implemented and should be preserved.
- Session invalidation relies on the `session_start` timestamp stored in both the JWT and the database.
- Authentication logic is intentionally modular; avoid merging the auth files into a single module unless explicitly requested.



---

# Authentication Verification

The authentication system has been manually verified through end-to-end testing.

The following authentication scenarios were successfully tested:

### Login

- User authentication with valid credentials
- Invalid credential handling
- Session initialization after successful login

### Access Token Renewal

- Access token expiration
- Automatic token refresh through the Axios interceptor
- Automatic retry of the original protected request
- Seamless continuation of the authenticated session

### Refresh Token Expiry

- Session expiration after refresh token timeout
- Automatic redirect to the login page
- Correct handling of missing or expired refresh tokens

### Absolute Session Expiry

- Verification that refresh token rotation cannot extend the maximum session lifetime
- Forced re-authentication after absolute session expiration

### Logout

- Refresh token invalidation
- Session invalidation
- Authentication cookie removal
- Protected routes becoming inaccessible after logout

### Password Change

- Current password verification
- Password update
- Immediate invalidation of all existing sessions
- Forced login with the new password

### Session Verification

Verified that protected routes correctly:

- Validate the current session
- Synchronize frontend authentication state
- Redirect unauthenticated users to the login page

All authentication flows behaved as expected during end-to-end manual testing.