# ── RBAC (Role-Based Access Control) ──────────────────────────────────────────
# This module defines who can access what data folders
# and provides the enforce_rbac() function to validate roles.
# ──────────────────────────────────────────────────────────────────────────────

# ── Step 1: Define who can access what ────────────────────────────────────────
# This dictionary maps each role to the folders they are allowed to read from.
# For example, "finance" can only read from the "finance" folder.
# "c-level" can read from ALL folders.

ROLE_FOLDERS = {
    "finance":     ["finance"],
    "hr":          ["hr", "general"],   # HR needs handbook for policy questions
    "engineering": ["engineering"],
    "marketing":   ["marketing"],
    "employee":    ["general"],         # employees only see the handbook
    "c-level":     ["finance", "hr", "engineering", "marketing", "general"],
}

# A set of all valid role names — used for fast lookup
# set() gives O(1) lookup — much faster than checking a list
VALID_ROLES = set(ROLE_FOLDERS.keys())

# Build a reverse map: folder → which roles are allowed in it
# Example: {"finance": {"finance", "c-level"}, "general": {"employee", "hr", "c-level"}}
# This is used in Layer 2 of RBAC to prevent privilege escalation
FOLDER_ALLOWED_ROLES = {}
for _role, _folders in ROLE_FOLDERS.items():
    for _folder in _folders:
        # setdefault creates an empty set if key doesn't exist, then adds the role
        FOLDER_ALLOWED_ROLES.setdefault(_folder, set()).add(_role)


# ── Step 2: RBAC — check if the user is allowed ───────────────────────────────
# RBAC = Role Based Access Control
# This function checks two things:
#   Layer 1: Is the role a valid known role?
#   Layer 2: Is the role actually allowed in the folders it wants to access?

def enforce_rbac(role: str) -> tuple:
    # Layer 1: basic checks
    # isinstance() checks if role is actually a string (not None or a number)
    if not isinstance(role, str) or role.strip() == "":
        return False, "Role must be a non-empty string."

    # remove spaces and make lowercase so "Finance " == "finance"
    role_clean = role.strip().lower()

    # Check if role exists in our allowed roles
    if role_clean not in VALID_ROLES:
        return False, f"Access denied: '{role_clean}' is not a recognised role. Valid roles: {sorted(VALID_ROLES)}."

    # Layer 2: escalation check
    # Even if role is valid, verify it's actually allowed in each folder it claims
    # This prevents someone from sneaking extra folder access at runtime
    for folder in ROLE_FOLDERS.get(role_clean, []):
        if role_clean not in FOLDER_ALLOWED_ROLES.get(folder, set()):
            return False, f"Access denied: role '{role_clean}' is not permitted to access folder '{folder}'."

    # Both checks passed — role is valid and permitted
    return True, ""
