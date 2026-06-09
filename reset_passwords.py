import json
import os
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

USERS_FILE = os.path.join(os.path.dirname(__file__), "backend", "users.json")

def load_users():
    with open(USERS_FILE, "r") as file:
        return json.load(file)

def save_users(users):
    with open(USERS_FILE, "w") as file:
        json.dump(users, file, indent=4)

def reset_all_passwords():
    users = load_users()
    for username, details in users.items():
        new_password = f"{username}123"
        details["password_hash"] = pwd_context.hash(new_password)
        details["failed_attempts"] = 0
        details["lock_until"] = 0
        details["lock_count"] = 0
        details["last_failed_login"] = 0
        print(f"Reset password for {username} to {new_password}")
    
    save_users(users)
    print("All passwords have been reset successfully.")

if __name__ == "__main__":
    reset_all_passwords()
