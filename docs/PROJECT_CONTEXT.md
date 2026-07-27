# PROJECT_CONTEXT.md

# Role-Based AI Assistant

Version: v2.0

---

# Note for AI Assistants

Treat this document as the authoritative description of the project architecture.

The backend architecture, authentication system, endpoint contracts, and frontend structure are considered stable. Preserve the existing architecture unless explicitly requested to redesign it.

Prefer extending the current modular implementation over introducing new frameworks or major architectural changes.

---

# Project Overview

Role-Based AI Assistant is a secure enterprise Retrieval-Augmented Generation (RAG) system designed to demonstrate enterprise-grade AI architecture, secure authentication, and Role-Based Access Control (RBAC).

The application enables employees to securely query internal company documents using a Large Language Model while ensuring that every response is generated only from documents the authenticated user is authorized to access.

Unlike traditional chatbots, authorization is enforced before document retrieval begins, ensuring complete separation between authentication, authorization, and AI response generation.

The project emphasizes:

- Secure Authentication
- Enterprise Session Management
- Role-Based Authorization (RBAC)
- Retrieval-Augmented Generation (RAG)
- Modular Backend Architecture
- Clean React Frontend Architecture
- Secure Cookie-Based Authentication
- Maintainable and Scalable Design

---

# Technology Stack

## Backend

- FastAPI
- PostgreSQL
- Psycopg2 Connection Pool
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- Qwen 3.6 27B
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

# High-Level Architecture

React Frontend
│
▼
Axios Services
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
└──────── Groq API (Qwen 3.6 27B)




---

# Authentication System

Authentication is entirely cookie-based.

The frontend never stores JWTs inside localStorage or sessionStorage.

The backend manages authentication using:

- HttpOnly Access Token Cookie
- HttpOnly Refresh Token Cookie

The browser automatically includes these cookies on authenticated requests.

Frontend communicates with the backend using:
withCredentials: true




---

# Authentication Features

Implemented security features include:

- JWT Access Tokens
- Refresh Tokens
- Refresh Token Rotation
- Refresh Tokens stored only as SHA-256 hashes
- HttpOnly Cookies
- Argon2 Password Hashing
- Automatic Access Token Refresh
- Absolute Session Expiry
- Session Binding using session_start
- Account Lockout
- Exponential Lock Duration
- Password Change Session Invalidation
- Logout Session Invalidation

Authentication state is managed entirely by the backend.

---

# Session Management

### Access Token

- Short-lived JWT
- Used for every authenticated request
- Automatically refreshed when expired

### Refresh Token

- Longer lifetime
- Stored only as SHA-256 hash inside PostgreSQL
- Rotated on every successful refresh
- Never accessible from frontend JavaScript

### Absolute Session

Every login creates a fixed maximum session lifetime.

Refresh token rotation extends access tokens but never extends the absolute session.

Once the maximum session duration is reached, re-authentication is required.

---

# Authorization (RBAC)

Every authenticated user belongs to exactly one application role.

Supported roles:

- finance
- hr
- engineering
- marketing
- employee
- c-level
- admin

Each role maps to predefined document folders.

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

Admin

→ Application Administration Only

The admin role intentionally manages users but is excluded from RAG document retrieval.

Authorization is always enforced before retrieval begins.

Role information is never trusted from the JWT and is always loaded from PostgreSQL.

---

# RAG Pipeline

Every AI request follows the pipeline below:

1. Authenticate User
2. Verify Session
3. Load Current User
4. Read Role from Database
5. Validate RBAC Permissions
6. Load Accessible Vector Databases
7. Perform Semantic Search using MMR
8. Build Context
9. Query Groq (Qwen 3.6 27B)
10. Return Answer with Sources

Only authorized documents participate in retrieval.

---

# Vector Database

Separate ChromaDB vector stores are maintained according to RBAC permissions.

Examples:
finance
engineering
marketing
hr
general





Document fingerprints are used to detect modifications and automatically rebuild vector databases when required.

---

# Embeddings

Embedding model:
all-MiniLM-L6-v2




The embedding model is loaded once using a Singleton pattern to minimize initialization overhead.

---

# Retrieval Strategy

Semantic retrieval uses:

Maximal Marginal Relevance (MMR)

Configuration:

- fetch_k = 30
- k = 10

This improves diversity while reducing redundant document chunks.

---

# Semantic Collision Prevention

To reduce semantic collisions between similar documents, metadata is prepended before embedding.

Metadata includes:

- Document Name
- Document Type
- Department
- Quarter
- Year

This improves retrieval accuracy for documents with similar content.

---

# Database

Primary Database:

PostgreSQL

Responsibilities:

- Users
- Roles
- Password Hashes
- Refresh Token Hashes
- Failed Login Attempts
- Account Lock Information
- Session Start Timestamp

Connection Pool:

psycopg2 SimpleConnectionPool

Every backend module shares the same centralized connection pool.

---

# Frontend Architecture

The frontend follows the architecture below:
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





Responsibilities:

Pages

- UI
- User Interaction

Hooks

- Session Verification

Context

- Authentication State

Services

- API Communication

Axios

- Cookie-Based Authentication
- Automatic Refresh
- Request Retry

Networking remains centralized inside the services layer.

---

# Frontend Features

Implemented functionality includes:

- Login
- Logout
- Dashboard
- Chat
- Home Dataset Viewer
- Dashboard Dataset Viewer (RBAC Protected)
- Set Password
- Change Password
- Admin Create User
- Protected Routes
- Session Verification
- Automatic Token Refresh
- Per-user Chat History
- Chat Persistence
- Document Viewing
- Authentication Context

The remaining frontend work focuses primarily on UI/UX improvements and responsive styling.

---

# API Communication

Axios automatically:

- Sends HttpOnly Cookies
- Refreshes Expired Access Tokens
- Retries Failed Requests
- Prevents Concurrent Refresh Requests

Frontend pages never communicate directly with Axios.

---

# Current Project Status

Backend

✅ Complete

Authentication

✅ Complete

Authorization (RBAC)

✅ Complete

RAG Pipeline

✅ Complete

PostgreSQL Integration

✅ Complete

Frontend Routing

✅ Complete

Authentication Context

✅ Complete

Protected Routes

✅ Complete

Dashboard

✅ Complete

Dataset Viewer

✅ Complete

Chat

✅ Complete

Login

✅ Complete

Set Password

✅ Complete

Change Password

✅ Complete

Admin User Creation

✅ Complete

End-to-End Authentication Testing

✅ Complete

End-to-End Functional Testing

✅ Complete


Remaining work below is also completed:
- UI/UX Styling       
- Responsive Design
- Visual Polish
- Deployment

---

# Coding Principles

This project intentionally follows a modular architecture.

Core design principles:

- Authentication and Authorization remain independent.
- Never trust roles stored inside JWTs.
- Always use HttpOnly Cookies.
- Never store JWTs inside localStorage or sessionStorage.
- Networking belongs inside the Services layer.
- Authentication state belongs inside Context.
- Pages primarily contain UI logic.
- Preserve modular backend components.
- Keep API communication centralized.
- Prefer readability and maintainability over unnecessary abstractions.

# Deployment

Backend
- AWS EC2
- Docker

Frontend
- React build served by Caddy

Database
- Supabase PostgreSQL

HTTPS
- Let's Encrypt

Domain
- DuckDNS

---

# Purpose of this Document

This document provides a high-level overview of the complete Role-Based AI Assistant architecture.

It is intended to help developers and AI assistants quickly understand the project's structure, authentication model, authorization strategy, frontend architecture, and retrieval pipeline without reading the entire codebase.

Detailed information regarding endpoints, authentication flow, folder structure, frontend implementation, and system architecture is documented separately within the `docs/` directory.


