from jose import jwt

secret = "mysecretkey"

token = jwt.encode({"user": "sid"}, secret, algorithm="HS256")
print(token)

data = jwt.decode(token, secret, algorithms=["HS256"])
print(data)