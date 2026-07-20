from fastapi import Request
from .config import (ABSOLUTE_SESSION_EXPIRE_DAYS )
import time

from .jwt_handler import verify_token
from .database import get_user


# Returns the currently authenticated user using the access token cookie
def get_current_user(request: Request):

    # Read the access token sent automatically by the browser
    access_token = request.cookies.get("access_token")

    # Cookie missing
    if not access_token:
        return None

    # Verify JWT
    payload = verify_token(access_token)

    # Invalid or expired JWT
    if not payload:
        return None

    # Extract employee ID from JWT payload
    emp_id = payload["sub"].lower()

    # Fetch latest user information from the database
    user = get_user(emp_id)

    # User no longer exists
    if not user:
        return None

    # -----------------------------
    # Session validation
    # -----------------------------

    # Absolute session expiry
    ABSOLUTE_SESSION_EXPIRE_SECONDS = ABSOLUTE_SESSION_EXPIRE_DAYS * 86400

    if time.time() > user.get("session_start", 0) + ABSOLUTE_SESSION_EXPIRE_SECONDS:
        return None

    # Session invalidated (logout/password change)
    token_session = payload.get("session_start")

    if token_session != user.get("session_start"):
        return None

    # Account locked
    if user["lock_until"] > time.time():
        return None

    return user