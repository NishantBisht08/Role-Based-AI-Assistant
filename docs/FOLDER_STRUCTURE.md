# Folder Structure

This document describes the directory structure of the Role-Based AI Assistant project and the responsibility of each folder/module.

---

# Project Structure

```text
Role-Based-AI-Assistant/
│
├── backend/
│   ├── auth/
│   ├── rag_engine/
│   ├── shared_cons.py
│   └── main.py
│
├── frontend/
│   ├── src/
│   │   ├── assets/
│   │   ├── components/
│   │   ├── contexts/
│   │   ├── hooks/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│
├── data/
│   ├── engineering/
│   ├── finance/
│   ├── hr/
│   ├── marketing/
│   └── general/
│
├── docs/
│
└── .env
```

---

# Backend

The backend is built using FastAPI and is responsible for authentication, authorization, retrieval, and AI response generation.

## auth/

Handles the complete authentication system.

Responsibilities:

- Login
- Logout
- JWT creation
- JWT verification
- Refresh token generation
- Refresh token rotation
- Password hashing
- Password setup
- Password change
- Current user authentication
- Database operations related to authentication

Files:

```
auth/
│
├── config.py
├── database.py
├── login.py
├── logout.py
├── password.py
├── refresh.py
├── jwt_handler.py
├── auth_guard.py
└── __init__.py
```

---

## rag_engine/

Implements the Retrieval-Augmented Generation (RAG) pipeline.

Responsibilities:

- Role-Based Access Control (RBAC)
- Loading documents
- Creating embeddings
- Building/loading ChromaDB
- Semantic search using MMR
- Prompt construction
- Calling the Groq LLaMA model

Files:

```
rag_engine/
│
├── rbac.py
├── document_loader.py
├── embeddings.py
├── vectorstore.py
├── pipeline.py
└── __init__.py
```

---

## shared_cons.py

Creates a shared PostgreSQL connection pool.

Every module that accesses the database imports the connection pool from this file.

---

## main.py

Entry point of the FastAPI application.

Responsibilities:

- API endpoints
- Cookie handling
- Session validation
- Calling authentication modules
- Calling the RAG pipeline

---

# Frontend

Built using React + Vite.

The frontend intentionally focuses on functionality before styling.

---

## assets/

Contains static resources.

Examples:

- Images
- Icons
- Videos
- Audio

---

## components/

Reusable UI components shared across multiple pages.

Examples:

- Navbar
- Footer
- ProtectedRoute
- ChatBox
- Message
- Example Questions

---

## contexts/

Global application state.

Currently contains:

- Authentication Context

Responsible for:

- Current logged-in user
- Authentication loading state
- User updates after login/logout

---

## hooks/

Reusable custom React hooks.

Current hook:

- useVerifySession()

Purpose:

- Verify the current session
- Synchronize AuthContext with backend authentication state

---

## pages/

Application pages.

Current pages include:

- Home
- Login
- Dashboard
- Chat
- About
- About Author
- Dataset
- Demo Guide
- Set Password
- Change Password

Each page is responsible only for UI and user interaction.

Networking is delegated to the services layer.

---

## services/

Responsible for backend communication.

Current responsibilities:

- Login
- Logout
- Current user
- Refresh token
- Ask Question

Also contains the shared Axios instance and refresh-token interceptor.

---

## App.jsx

Defines all application routes.

Responsible for:

- React Router
- Protected routes
- Shared Navbar
- Shared Footer

---

## main.jsx

Application entry point.

Responsible for:

- Rendering React
- Loading global CSS
- Wrapping the application with AuthProvider

---

# data/

Contains enterprise documents used by the RAG system.

Documents are separated by department.

Example:

```
finance/
engineering/
marketing/
hr/
general/
```

These folders are indexed into separate ChromaDB vector stores based on RBAC permissions.

---

# docs/

Project documentation.

Current documents:

- PROJECT_CONTEXT.md
- API_REFERENCE.md
- AUTH_FLOW.md
- FOLDER_STRUCTURE.md
- FRONTEND_STATUS.md

These files exist to quickly provide architectural context without reading the complete codebase.

---

# Design Principle

The project follows a modular architecture.

Each folder has a single responsibility.

Business logic, authentication, networking, routing, retrieval, and presentation are intentionally separated to improve maintainability and scalability.