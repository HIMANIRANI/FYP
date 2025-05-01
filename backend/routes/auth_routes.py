import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from google.auth.transport import requests
from google.oauth2 import id_token
from motor.motor_asyncio import AsyncIOMotorClient

from ..configurations.config import settings
from ..models.token_model import LoginRequest, Token, UserInfo
from ..models.user_model import User
from ..services.auth_services import (create_access_token, hash_password,
                                      verify_password, verify_token)

# Initialize the FastAPI router
router = APIRouter()

# Initialize the MongoDB client and define the database and collection
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata  # Database for user-related data
collection_name = db["users"]  # Collection to store user information

# Define the OAuth2 password bearer token scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

IMAGEDIR = "profile_pictures/"
os.makedirs(IMAGEDIR, exist_ok=True)  # Ensure the directory exists


@router.post("/register")
async def register_user(user: User):
    """
    Endpoint to register a new user.
    - Checks if passwords match
    - Verifies if the email is already in use
    - Hashes the password and stores user data in MongoDB
    - Assigns a default profile image
    """
    if user.password != user.confirmPassword:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    user_db = await collection_name.find_one({"email": user.email})
    if user_db:
        raise HTTPException(status_code=400, detail="User already exists")

    # Hash the password before storing it in the database
    hashed_password = hash_password(user.password)
    new_user = {
        "firstName": user.firstName,
        "lastName": user.lastName,
        "email": user.email,
        "password": hashed_password,  # Store the hashed password
        "profile_image": f"{IMAGEDIR}default.jpg",  # Assign a default profile picture
    }

    # Insert the new user into the database
    result = await collection_name.insert_one(new_user)
    return {"message": "Account created successfully!", "user_id": str(result.inserted_id)}


@router.post("/auth/login", response_model=Token)
async def login(request: LoginRequest):
    """
    Endpoint to authenticate a user.
    - Checks if the user exists in the database
    - Verifies the password
    - Generates and returns an access token if authentication is successful
    """
    user = await collection_name.find_one({"email": request.email})
    if not user:
        raise HTTPException(status_code=400, detail="User not found.")
    
    # Verify the provided password against the stored hashed password
    if not verify_password(request.password, user["password"]):
        raise HTTPException(status_code=400, detail="Incorrect password.")
    
    # Generate an access token
    access_token = create_access_token(data={"sub": user["email"]})
    
    # Prepare user profile data (excluding sensitive information)
    user_profile = {
        "email": user["email"],
        "firstName": user.get("firstName", ""),
        "lastName": user.get("lastName", ""),
        "profile_image": user.get("profile_image", ""),
    }
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user_profile
    }


@router.get("/protected", response_model=UserInfo)
async def protected_route(token: str = Depends(oauth2_scheme)):
    """
    Protected route that requires authentication.
    - Extracts and verifies the token
    - Returns user information if the token is valid
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify the provided access token
    token_data = verify_token(token, credentials_exception)
    
    # Return the decoded token data (e.g., username and expiration time)
    return {"username": token_data.username, "exp": token_data.exp}


@router.post("/auth/google-login", response_model=Token)
async def google_login(data: dict):
    """
    Endpoint for logging in with Google OAuth.
    - Validates the Google token
    - If user does not exist, registers them
    - Returns a JWT access token
    """
    google_token = data.get("credential")
    if not google_token:
        raise HTTPException(status_code=400, detail="Google token is missing")

    try:
        # Verify Google token
        google_user = id_token.verify_oauth2_token(
            google_token, requests.Request(), settings.GOOGLE_CLIENT_ID
        )
        email = google_user.get("email")
        first_name = google_user.get("given_name", "")
        last_name = google_user.get("family_name", "")

        if not email:
            raise HTTPException(status_code=400, detail="Invalid Google token")

        # Check if the user already exists in the database
        user = await collection_name.find_one({"email": email})

        if not user:
            # Register the user if they do not exist
            new_user = {
                "firstName": first_name,
                "lastName": last_name,
                "email": email,
                "password": None,  # No password since they logged in via Google
                "profile_image": f"{IMAGEDIR}default.jpg",  # Assign a default profile picture
            }
            await collection_name.insert_one(new_user)
            user = new_user

        # Generate an access token for the user
        access_token = create_access_token(data={"sub": email})
        
        # Prepare user profile data
        user_profile = {
            "email": user["email"],
            "firstName": user.get("firstName", ""),
            "lastName": user.get("lastName", ""),
            "profile_image": user.get("profile_image", ""),
        }
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_profile
        }

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid Google token")


@router.get("/api/profile/get")
async def get_profile(token: str = Depends(oauth2_scheme)):
    """
    Endpoint to get user profile information.
    - Verifies the token
    - Returns user profile data
    """
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # Verify the provided access token
    token_data = verify_token(token, credentials_exception)
    
    # Get user data from database
    user = await collection_name.find_one({"email": token_data.username})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Return user profile data (excluding sensitive information)
    return {
        "email": user["email"],
        "firstName": user.get("firstName", ""),
        "lastName": user.get("lastName", ""),
        "profile_image": user.get("profile_image", ""),
    }
