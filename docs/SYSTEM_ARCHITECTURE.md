# SYSTEM_ARCHITECTURE.md

# System Architecture

Version: v1.0

This document describes the end-to-end architecture of the Role-Based AI Assistant, including request flow, authentication, authorization, retrieval pipeline, frontend interaction, and system components.

---

# High-Level Architecture

```
                            +----------------------+
                            |        User          |
                            +----------+-----------+
                                       |
                                       | HTTPS
                                       |
                            +----------v-----------+
                            | DuckDNS Domain       |
                            | novaris-rag.duckdns.org
                            +----------+-----------+
                                       |
                                       |
                            +----------v-----------+
                            | AWS EC2 Instance     |
                            +----------+-----------+
                                       |
                 +---------------------+----------------------+
                 |                                            |
                 |                                            |
      +----------v-----------+                     +-----------v----------+
      |       Caddy          |                     |   React Frontend     |
      | Reverse Proxy        |<------------------->|   (Built dist/)      |
      +----------+-----------+                     +----------------------+
                 |
                 | Reverse Proxy
                 |
      +----------v-----------+
      | Docker Container     |
      | FastAPI Backend      |
      +----------+-----------+
                 |
      +----------+-----------+------------------------------+
      |                      |                              |
      |                      |                              |
+-----v------+      +--------v--------+          +----------v----------+
|Authentication|     | Supabase       |          |     RAG Engine      |
+-------------+     | PostgreSQL DB  |          +----------+----------+
                                                 |
                                      +----------v----------+
                                      |     ChromaDB        |
                                      +----------+----------+
                                                 |
                                      +----------v----------+
                                      | HuggingFace         |
                                      | Embedding Model     |
                                      +----------+----------+
                                                 |
                                      +----------v----------+
                                      |   Groq Qwen 3.6     |
                                      |      27B LLM        |
                                      +---------------------+
```

---

# Request Flow

Every user request follows this path.

```
User

↓

HTTPS Request

↓

DuckDNS

↓

AWS EC2

↓

Caddy

↓

React Frontend

↓

Axios

↓

FastAPI

↓

Authentication

↓

Authorization (RBAC)

↓

RAG Pipeline

↓

ChromaDB

↓

Groq Qwen 3.6 27B

↓

Response

↓

React Frontend

↓

User
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

Argon2 Password Verification

↓

Generate Access Token

↓

Generate Refresh Token

↓

Store Refresh Token Hash

↓

Set HttpOnly Cookies

↓

Dashboard
```

---

# Authenticated Request Flow

```
Frontend

↓

GET /me

↓

Read Access Token Cookie

↓

Verify JWT

↓

Load User

↓

Verify Session

↓

Return User
```

---

# AI Query Flow

```
User enters question

↓

POST /ask

↓

Read Access Token Cookie

↓

Verify JWT

↓

Load User

↓

Absolute Session Check

↓

Session Binding Check

↓

Account Lock Check

↓

Load Role

↓

RBAC

↓

Load ChromaDB

↓

MMR Retrieval

↓

Prompt Construction

↓

Groq LLaMA

↓

Answer

↓

Frontend
```

---

# Automatic Token Refresh

```
Protected Request

↓

401 Unauthorized

↓

Axios Interceptor

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

↓

Success
```

Only one refresh request is allowed at a time.

Concurrent failed requests reuse the same refresh operation.

---

# Logout Flow

```
POST /logout

↓

Delete Refresh Token

↓

Reset Session

↓

Clear Cookies

↓

Frontend Redirect
```

---

# Password Change Flow

```
Authenticated User

↓

Verify Current Password

↓

Hash New Password

↓

Update Database

↓

Delete Refresh Token

↓

Reset Session

↓

Force Login
```

# Admin User Creation Flow


Authenticated Admin

↓

POST /admin/create-user

↓

Verify JWT

↓

Load Current User

↓

Verify Admin Role

↓

Validate Employee Data

↓

Insert User

↓

Return Success


---

# RAG Pipeline

```
Question

↓

RBAC

↓

Determine Accessible Folders

↓

Load ChromaDB

↓

Semantic Search

↓

MMR

↓

Top Chunks

↓

Prompt Builder

↓

Groq

↓

Answer
```

---

# Authentication Components

```
Frontend

↓

HttpOnly Cookies

↓

verify_token()

↓

Database Lookup

↓

Session Validation

↓

Authorization
```

---

# Database Responsibilities

PostgreSQL stores:

- Users
- Password Hashes
- Roles
- Refresh Token Hashes
- Failed Login Attempts
- Lock Timers
- Session Start Timestamp

ChromaDB stores:

- Document Embeddings
- Metadata
- Vector Indexes

---

# Frontend Architecture

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

Backend
```

Responsibilities:

Pages

- UI
- User interaction

Hooks

- Session helpers

Context

- Authentication state

Services

- API communication

Axios

- HTTP transport
- Token refresh

---

# Backend Architecture

```
main.py

↓

Authentication

↓

Authorization

↓

Database

↓

RAG Engine

↓

Response
```

Authentication modules remain independent from RAG modules.

---

# Security Architecture

Security layers execute in this order.

```
Cookie Exists

↓

JWT Valid

↓

User Exists

↓

Absolute Session

↓

Session Binding

↓

Account Lock

↓

Role Lookup

↓

Application Authorization

↓

RBAC

↓

Document Retrieval
```

A request must pass every layer before reaching the LLM.

---

# Design Principles

The architecture follows these principles:

- Modular design
- Separation of concerns
- Stateless authentication with server-side validation
- Database as the source of truth
- Least-privilege access through RBAC
- Shared database connection pooling
- Automatic token refresh
- Secure cookie-based authentication
- Centralized API communication
- Reusable frontend components
- Administrative privileges are independent of document-access permissions.

---

# Scalability Considerations

Current Version

- Single FastAPI instance
- PostgreSQL
- ChromaDB
- Groq API

Future Improvements

- Redis caching
- Multiple FastAPI workers
- Load balancer
- Docker deployment
- Kubernetes orchestration
- Background document indexing
- Horizontal scaling
- Streaming LLM responses

---

# End-to-End Summary

```
User

↓

React UI

↓

Axios

↓

FastAPI

↓

Authentication

↓

PostgreSQL

↓

RBAC

↓

ChromaDB

↓

Groq LLaMA

↓

Answer

↓

Frontend
```

This architecture emphasizes security, modularity, maintainability, and enterprise-ready authentication while providing role-aware Retrieval-Augmented Generation for internal organizational documents.



# short summary

                    USER
                      │
                      ▼
               React Frontend
                      │
                Axios Services
                      │
                      ▼
               FastAPI Backend
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
 Authentication   PostgreSQL DB    RAG Engine
      │                                │
      │                                ▼
      │                           ChromaDB
      │                                │
      └──────────────────────┐         ▼
                             ▼
                        Groq LLaMA
                             │
                             ▼
                      AI Response
                             │
                             ▼
                       React Frontend