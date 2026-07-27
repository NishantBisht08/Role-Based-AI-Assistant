# DEPLOYMENT.md

# Novaris Deployment Guide

Version: v1.0

This document describes the complete deployment process for Novaris, including backend deployment, frontend deployment, Docker configuration, HTTPS setup, and deployment updates.

---

# Deployment Architecture

```
User

↓

DuckDNS Domain

↓

AWS EC2 Instance

↓

Caddy

├───────────────┐
│               │
│ React Frontend│
│ (dist folder) │
│               │
└───────────────┘

↓

Docker Container

↓

FastAPI Backend

↓

Supabase PostgreSQL

↓

Groq API
```

---

# Technology Used

Infrastructure

- AWS EC2 (Ubuntu 24.04)
- Docker
- Caddy
- DuckDNS
- Let's Encrypt

Backend

- FastAPI
- Docker

Frontend

- React + Vite

Database

- Supabase PostgreSQL

AI

- Groq API
- ChromaDB
- HuggingFace Embeddings

---

# Deployment Workflow

Development

```
Local Development

↓

Git Commit

↓

Git Push

↓

EC2

↓

Git Pull

↓

Docker Build

↓

Docker Run

↓

Production
```

---

# Backend Deployment

## 1. Connect to EC2

```bash
ssh -i "novaris-key.pem" ubuntu@<PUBLIC_IP>
```

---

## 2. Navigate to project

```bash
cd ~/Role-Based-AI-Assistant
```

---

## 3. Pull latest code

```bash
git pull
```

---

## 4. Build Docker Image

```bash
sudo docker build -t novaris-backend .
```

---

## 5. Stop Existing Container

```bash
sudo docker stop novaris-backend
```

---

## 6. Remove Existing Container

```bash
sudo docker rm novaris-backend
```

---

## 7. Start New Container

```bash
sudo docker run -d \
--name novaris-backend \
-p 8000:8000 \
--env-file backend/.env \
novaris-backend
```

---

# Frontend Deployment

## Build Production Version

```bash
npm run build
```

This generates the production-ready `dist/` folder.

---

## Copy dist to EC2

```bash
scp -i "novaris-key.pem" -r frontend/dist ubuntu@<PUBLIC_IP>:~
```

---

## Copy Files to Caddy

```bash
sudo cp -r ~/dist/* /var/www/html/
```

Caddy immediately serves the updated frontend.

---

# Caddy

Caddy is responsible for:

- Serving the React frontend
- HTTPS
- Reverse Proxy
- TLS certificate management

API requests are forwarded to:

```
localhost:8000
```

where the FastAPI backend is running.

---

# DuckDNS

DuckDNS maps

```
novaris-rag.duckdns.org
```

to the EC2 public IP.

If the EC2 public IP changes after stopping the instance, only the DuckDNS IP needs updating.

---

# HTTPS

HTTPS is automatically provided by:

- Caddy
- Let's Encrypt

No manual certificate management is required.

---

# Updating the Application

Backend Changes

```
git push

↓

EC2

↓

git pull

↓

docker build

↓

docker stop

↓

docker rm

↓

docker run
```

Frontend Changes

```
npm run build

↓

Copy dist

↓

Replace files inside

/var/www/html
```

---

# Monitoring

Check Disk Usage

```bash
df -h
```

Check RAM

```bash
free -h
```

Live Container Memory

```bash
sudo docker stats
```

Docker Disk Usage

```bash
sudo docker system df
```

Running Containers

```bash
sudo docker ps
```

All Containers

```bash
sudo docker ps -a
```

Docker Images

```bash
sudo docker images
```

---

# Cleanup

Remove unused Docker images

```bash
sudo docker image prune -a
```

Remove stopped containers

```bash
sudo docker container prune
```

Remove unused Docker resources

```bash
sudo docker system prune -a
```

---

# EC2 Stop / Resume

Stopping an EC2 instance:

- Preserves Ubuntu
- Preserves Docker
- Preserves Caddy
- Preserves Project Files
- Preserves Docker Images
- Preserves Docker Containers

If the public IP changes after restart:

1. Update DuckDNS.
2. Verify Caddy is running.
3. Verify Docker container is running.

---

# Environment Variables

Backend

Stored on EC2:

```
backend/.env
```

Contains:

- DATABASE_URL
- GROQ_API_KEY
- SECRET_KEY
- CLIENT_URL

This file is not committed to Git.

Frontend

Development

```
.env.development
```

Production

```
.env.production
```

Vite automatically selects the correct file.

---

# Useful Commands

Connect to EC2

```bash
ssh -i "novaris-key.pem" ubuntu@<PUBLIC_IP>
```

Restart Docker Container

```bash
sudo docker restart novaris-backend
```

Check Docker Logs

```bash
sudo docker logs novaris-backend
```

Restart Caddy

```bash
sudo systemctl restart caddy
```

Check Caddy Status

```bash
sudo systemctl status caddy
```

Check Docker Status

```bash
sudo systemctl status docker
```

---

# Deployment Summary

The deployment process consists of:

1. Deploy backend to AWS EC2 using Docker.
2. Deploy frontend by serving the built React files through Caddy.
3. Secure the application using HTTPS with Let's Encrypt.
4. Map a DuckDNS domain to the EC2 instance.
5. Connect the backend to Supabase PostgreSQL.
6. Update deployments through Git, Docker rebuilds, and frontend rebuilds.

The deployed application is accessible securely from any internet-connected device through the DuckDNS domain.