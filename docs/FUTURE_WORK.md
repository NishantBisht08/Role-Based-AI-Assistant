# Future Work

The current implementation focuses on delivering a complete Role-Based AI Assistant with secure authentication, RBAC, Retrieval-Augmented Generation (RAG), and a fully functional frontend. The following enhancements are planned for future iterations of the project.

---

## 1. Redis Integration

### Objective

Replace selected database operations with Redis to improve performance and scalability.

### Planned Implementation

Redis will be introduced as an in-memory data store alongside PostgreSQL.

Potential use cases include:

- Session caching
- Refresh token storage
- Frequently accessed user information
- Temporary authentication data
- Rate limiting counters

Authentication flow would become:

```
Client

↓

FastAPI

↓

Redis (Cache)

↓

PostgreSQL (Persistent Storage)
```

### Benefits

- Faster authentication requests
- Reduced PostgreSQL load
- Lower database latency
- Better scalability for concurrent users
- Foundation for future distributed deployments

---

## 2. API Rate Limiting

### Objective

Protect the application against brute-force attacks and excessive API usage.

### Planned Implementation

Introduce request limits using Redis-backed counters.

Each endpoint can have independent limits, for example:

| Endpoint | Planned Limit |
|-----------|---------------|
| /login | 5 requests/minute |
| /refresh | 10 requests/minute |
| /ask | 20 requests/minute |
| Dataset APIs | Higher limits |

Rate limiting will be applied using:

- Client IP address
- Authenticated Employee ID (where applicable)

When limits are exceeded:

- HTTP 429 (Too Many Requests) will be returned.
- Requests will automatically resume after the configured cooldown period.

### Benefits

- Protection against brute-force login attempts
- Prevention of API abuse
- Reduced LLM usage costs
- Improved server stability
- Better resource utilization

---

## 3. Authentication Improvements

Although the current authentication system has been thoroughly tested and functions correctly, the following refinements are planned:

### Idempotent Logout

Modify the logout endpoint so that authentication cookies are cleared even if the refresh token has already been invalidated or rotated.

This improves logout robustness in multi-tab browser sessions.

### Integer Session Timestamps

Replace floating-point session timestamps with integer Unix timestamps.

This removes any dependency on floating-point equality comparisons between JWT payloads and database values.

---

## UI/UX Enhancements (Completed in v1)

Frontend improvements include:

- Fully responsive design
- Modern dashboard layout
- Improved chat interface
- Enhanced dataset viewer
- Loading animations
- Better visual feedback
- Accessibility improvements
- Dark mode support

---

## Deployment (Completed in v1)

✓ AWS EC2 deployment
✓ Dockerized backend
✓ Caddy reverse proxy
✓ HTTPS via Let's Encrypt
✓ DuckDNS custom domain
✓ Supabase PostgreSQL integration

Future deployment improvements

- CI/CD using GitHub Actions
- Docker Compose
- Nginx/Caddy configuration optimization
- Kubernetes (learning project)
- Monitoring and logging