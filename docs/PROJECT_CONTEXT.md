# PROJECT_CONTEXT.md

# Role-Based AI Assistant

Version: v1.0

# Note for AI Assistants

Treat this document as the authoritative description of the project architecture. Preserve the existing architecture and endpoint contracts unless explicitly asked to redesign them. Prefer extending the current modular structure over introducing new patterns or frameworks.

---

# Project Overview

Role-Based AI Assistant is a secure enterprise Retrieval-Augmented Generation (RAG) system built to demonstrate enterprise-grade AI architecture.

The application allows employees to securely query internal company documents using a Large Language Model while enforcing strict Role-Based Access Control (RBAC).

Unlike a traditional chatbot, every answer is generated only from documents that the authenticated user is authorized to access.

The project emphasizes:

- Secure authentication
- Authorization
- Enterprise session management
- Retrieval-Augmented Generation
- Modular backend architecture
- Clean React frontend architecture

---

# Tech Stack

## Backend

- FastAPI
- PostgreSQL
- Psycopg2 Connection Pool
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- LLaMA 3.3 70B
- JWT Authentication
- Argon2 Password Hashing

---

## Frontend

- React
- Vite
- React Router
- Axios
- Context API

---

# High Level Architecture

```
React Frontend
        │
        ▼
Axios API Layer
        │
        ▼
FastAPI Backend
        │
        ├──────── Authentication
        │
        ├──────── Authorization (RBAC)
        │
        ├──────── PostgreSQL
        │
        ├──────── ChromaDB
        │
        └──────── Groq LLaMA
```

---

# Authentication System

Authentication is entirely cookie based.

The frontend never stores JWTs inside localStorage or sessionStorage.

The backend sets:

- HttpOnly Access Token Cookie
- HttpOnly Refresh Token Cookie

The browser automatically sends these cookies on authenticated requests.

Frontend communicates using:

```
withCredentials: true
```

---

# Authentication Features

Implemented features include:

- JWT Access Tokens
- Refresh Tokens
- Refresh Token Rotation
- Refresh Tokens stored as SHA-256 hashes
- Argon2 password hashing
- HttpOnly cookies
- Automatic token refresh
- Absolute session expiry
- Account lockout
- Exponential lock duration
- Password change invalidates all sessions
- Logout invalidates refresh tokens
- Session binding using session_start timestamp

---

# Session Management

Access Token

- Short lifetime
- Used for every authenticated request

Refresh Token

- Longer lifetime
- Rotated every refresh request
- Stored only as SHA-256 hash inside PostgreSQL

Absolute Session

- Maximum session lifetime
- Cannot be extended by refresh token rotation

---

# Authorization (RBAC)

Every authenticated user belongs to exactly one role.

Supported roles:

- finance
- hr
- engineering
- marketing
- employee
- c-level
- admin

Each role is mapped to one or more document folders.

Examples:

Finance
→ finance/

HR
→ hr/
→ general/

Employee
→ general/

C-Level
→ finance/
→ hr/
→ engineering/
→ marketing/
→ general/

Authorization is enforced before document retrieval begins.

Role is NEVER trusted from the JWT.

The backend always reloads the user's current role from PostgreSQL.

---

# RAG Pipeline

The RAG pipeline performs the following steps:

1. Authenticate user
2. Verify session
3. Read role from database
4. Validate RBAC permissions
5. Load role-specific ChromaDB
6. Perform semantic search using MMR
7. Build prompt
8. Call Groq LLaMA
9. Return answer with source documents

---

# Vector Database

Separate ChromaDB databases are maintained per role.

Examples:

```
chroma_db_finance
chroma_db_hr
chroma_db_engineering
...
```

Vector databases are automatically rebuilt whenever source documents change.

File fingerprinting is used to detect changes.

---

# Embeddings

Embedding model:

```
all-MiniLM-L6-v2
```

Embedding model is loaded once using the Singleton pattern.

---

# Retrieval Strategy

Document retrieval uses:

Maximal Marginal Relevance (MMR)

Configuration:

- fetch_k = 30
- k = 10

This improves diversity of retrieved document chunks.

---

# Semantic Collision Prevention

Every document chunk is prepended with metadata before embedding.

Metadata includes:

- Document name
- Document type
- Quarter
- Year

This reduces collisions between semantically similar documents (for example Q1 vs Q4 reports).

---

# Database

Database:

PostgreSQL

Connection pooling:

psycopg2 SimpleConnectionPool

All modules share one centralized connection pool.

---

# Frontend Architecture

Frontend follows this structure:

```
Pages

↓

Hooks

↓

Context

↓

Services

↓

Axios

↓

FastAPI
```

Pages never communicate directly with Axios.

Networking is centralized inside the services layer.

Authentication state is centralized inside AuthContext.

---

# API Communication

Axios automatically:

- sends HttpOnly cookies
- refreshes expired access tokens
- retries failed requests after refresh

Only one refresh request may run at a time.

Concurrent requests share the same refresh operation.

---

# Current Implementation Status

Backend

✅ Complete

Authentication

✅ Complete

Authorization

✅ Complete

RAG Pipeline

✅ Complete

Frontend Routing

✅ Complete

Authentication Context

✅ Complete

Protected Routes

✅ Complete

Dashboard

✅ Complete

Chat

✅ Complete

Login

✅ Complete

Remaining frontend work:

- Set Password page
- Change Password page
- Admin page
- UI styling

---

# Coding Principles

This project intentionally follows a modular architecture.

Important design decisions:

- Keep backend modules separated.
- Never trust roles stored in JWT.
- Never store JWTs inside localStorage.
- Always use HttpOnly cookies.
- Networking belongs inside services/.
- Authentication state belongs inside Context.
- Pages should primarily contain UI logic.
- Avoid tightly coupling backend modules.

---

# Purpose of this Document

This document provides enough architectural context for another developer or AI assistant to understand the complete project without reading every source file.

Detailed information about endpoints, authentication flow, folder structure, and frontend implementation is documented separately inside the docs/ directory.