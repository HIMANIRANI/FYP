import os


class Settings:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://database:dbpassword@userdata.o9orl.mongodb.net/?retryWrites=true&w=majority&appName=userdata")
    SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "886481282340-ua5r107135v0lc58kngkgsb0tvvb2kii.apps.googleusercontent.com")

settings = Settings()
