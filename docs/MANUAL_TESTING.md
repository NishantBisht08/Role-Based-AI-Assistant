# Manual Testing Report

> **Status:** ✅ Completed
>
> This document records the manual end-to-end testing performed on the Role-Based AI Assistant.

---

# ✅ 1. Logged Out State

## Protected Pages

The following pages **cannot** be accessed directly by typing their URL:

- ✅ Dashboard
- ✅ Dashboard Dataset
- ✅ Chat
- ✅ Create User
- ✅ Change Password

## Public Pages

The following pages remain accessible:

- ✅ Home
- ✅ Dataset
- ✅ About
- ✅ Demo Guide
- ✅ Login
- ✅ Set Password

---

# ✅ 2. Set Password

### Password Mismatch

- ✅ Displays **"Password mismatch"**

### Invalid Employee ID

- ✅ Displays **"User not found"**

### Password Already Exists

- ✅ Displays **"Password already exists"**

### Valid First-Time User

Verified:

- ✅ Empty/space-only password is rejected.
- ✅ Password is successfully initialized.
- ✅ User is redirected to Login.

---

# ✅ 3. Login

### Invalid Employee ID

- ✅ Displays **"Invalid employee ID or password"**

### Password Not Initialized

- ✅ Displays **"Invalid employee ID or password"**

### Successful Login

- ✅ Redirects user to Dashboard.

---

# ✅ 4. Progressive Account Locking

Verified:

- ✅ Lock after 5 incorrect attempts.
- ✅ Lock duration increases exponentially.
- ✅ Failed attempts reset after 24 hours.
- ✅ Successful login resets failed-attempt counter.
- ✅ Lock occurs only for:
  - Correct Employee ID
  - Incorrect Password

---

# ✅ 5. Logged In State

After successful login:

Accessible pages:

- ✅ Dashboard
- ✅ Dashboard Dataset
- ✅ Chat
- ✅ Change Password

Public pages remain accessible.

Create User:

- ❌ Not accessible for normal users.
- ✅ Accessible only for Admin.

---

# ✅ 6. Chat

Verified:

- ✅ Maximum 2000-character input.
- ✅ Example Questions toggle works correctly.
- ✅ Chat history survives page navigation.
- ✅ Chat history survives browser refresh.
- ✅ Logout clears stored chat history.
- ✅ Oldest conversation is removed after exceeding 5 exchanges.
- ✅ Navbar updates Login → Logout correctly.
- ✅ Footer functions correctly.

---

# ✅ 7. Dashboard

Verified display of:

- ✅ Name
- ✅ Employee ID
- ✅ Role
- ✅ Accessible Documents

Verified actions:

- ✅ Start Chat
- ✅ View Dataset
- ✅ Change Password

Admin only:

- ✅ Create User button

---

# ✅ 8. Admin Functionality

Verified:

- ✅ Create User button visible.
- ✅ Create User page accessible through Dashboard.
- ✅ Direct URL access allowed for Admin.
- ✅ Admin has no RAG document access.
- ✅ Dataset correctly displays no accessible documents.

---

# ✅ 9. Create User

Verified:

- ✅ Duplicate Employee ID validation.
- ✅ Empty-field validation.
- ✅ Frontend prevents Admin creation.
- ✅ Role selection validation.

---

# ✅ 10. Dataset

## Public Dataset

Verified:

- ✅ All documents visible.
- ✅ Click to open document.
- ✅ Click again to close document.

## Dashboard Dataset

Verified:

- ✅ Only role-authorized documents displayed.
- ✅ RBAC enforced.
- ✅ Protected against unauthenticated access.

---

# ✅ 11. Session Management

## Access Token

- ✅ Automatic refresh after expiry.
- ✅ Original request automatically retried.

## Refresh Token

- ✅ Session expires after inactivity.
- ✅ User redirected to Login.

## Absolute Session

- ✅ Session expires after configured maximum lifetime.
- ✅ Re-authentication required.

Verified across:

- ✅ Dashboard
- ✅ Chat
- ✅ Dataset
- ✅ Change Password
- ✅ AI Question endpoint

---

# ✅ Overall Result

## Authentication

- ✅ Verified

## Authorization (RBAC)

- ✅ Verified

## Session Management

- ✅ Verified

## Password Management

- ✅ Verified

## Account Locking

- ✅ Verified

## Admin Functionality

- ✅ Verified

## Dataset Access

- ✅ Verified

## Chat Functionality

- ✅ Verified

## Route Protection

- ✅ Verified

## Token Refresh

- ✅ Verified

## Absolute Session Expiry

- ✅ Verified

---

# Final Status

**Overall Result:** ✅ **PASS**

All implemented functionality was manually verified through end-to-end testing.

No issues were observed during testing, and all tested features behaved as expected.