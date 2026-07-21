import secrets
print(secrets.token_hex(32))

#This function generates secret key which we store in .env file and used for jwt signing and verification