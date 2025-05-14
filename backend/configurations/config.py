import os


class Settings:
    MONGODB_URI = os.getenv("MONGODB_URI", "mongodb+srv://database:dbpassword@userdata.o9orl.mongodb.net/?retryWrites=true&w=majority&appName=userdata")
    SECRET_KEY = os.getenv("SECRET_KEY")
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    GOOGLE_CLIENT_ID: os.getenv("GOOGLE_CLIENT_ID")

settings = Settings()
