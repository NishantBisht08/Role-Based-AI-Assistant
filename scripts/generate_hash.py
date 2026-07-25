from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

passwords = {
    "admin": "admin@123"
}

for user, pwd in passwords.items():
    print(user, pwd_context.hash(pwd))
    
#In this file, we manually wrote  password for admin and generated hash for it and stored it manually in users.json file