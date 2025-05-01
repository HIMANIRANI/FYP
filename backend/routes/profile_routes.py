import os
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient

from ..configurations.config import settings
from ..services.auth_services import hash_password, verify_token

router = APIRouter()

client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata
collection_name = db["users"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

IMAGEDIR = "static/profile_pictures/"
os.makedirs(IMAGEDIR, exist_ok=True)  # Ensure the directory exists

# Copy default profile picture if it doesn't exist
DEFAULT_PROFILE_PIC = os.path.join(IMAGEDIR, "default.jpg")
if not os.path.exists(DEFAULT_PROFILE_PIC):
    import shutil
    default_source = os.path.join("backend", "static", "default_profile.jpg")
    shutil.copy(default_source, DEFAULT_PROFILE_PIC)

@router.put("/profile/update")
async def update_profile(
    firstName: Optional[str] = Form(None),
    lastName: Optional[str] = Form(None),
    email: Optional[str] = Form(None),
    password: Optional[str] = Form(None),
    profile_image: Optional[UploadFile] = File(None),
    token: str = Depends(oauth2_scheme),
):
    """Update user profile information, including profile image and password."""
    credentials_exception = HTTPException(
        status_code=401, detail="Invalid credentials"
    )
    
    token_data = verify_token(token, credentials_exception)
    user_email = token_data.username

    user = await collection_name.find_one({"email": user_email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = {}

    # Update basic information if provided
    if firstName is not None:
        update_data["firstName"] = firstName
    if lastName is not None:
        update_data["lastName"] = lastName
    if email is not None and email != user_email:
        existing_user = await collection_name.find_one({"email": email})
        if existing_user:
            raise HTTPException(status_code=400, detail="Email already in use")
        update_data["email"] = email
    if password is not None and password.strip():
        update_data["password"] = hash_password(password)

    # Handle profile image upload
    if profile_image:
        try:
            # Validate image file
            content_type = profile_image.content_type
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")
            
            # Generate unique filename and save
            ext = os.path.splitext(profile_image.filename)[1]
            filename = f"{uuid.uuid4()}{ext}"
            filepath = os.path.join(IMAGEDIR, filename)
            
            # Save new image
            content = await profile_image.read()
            with open(filepath, "wb") as f:
                f.write(content)

            # Remove old profile image if it exists and is not default
            old_image = user.get("profile_image")
            if old_image and "default.jpg" not in old_image:
                try:
                    os.remove(old_image)
                except (FileNotFoundError, OSError):
                    pass

            update_data["profile_image"] = filepath

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading image: {str(e)}")

    if update_data:
        result = await collection_name.update_one(
            {"email": user_email}, 
            {"$set": update_data}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="No changes made to profile")

        # Fetch updated user data
        updated_user = await collection_name.find_one(
            {"email": email if email else user_email},
            {"_id": 0, "password": 0}
        )
        return updated_user
    
    return user

@router.get("/profile/get")
async def get_profile(token: str = Depends(oauth2_scheme)):
    """Fetch the user's profile data, including profile image."""
    credentials_exception = HTTPException(status_code=401, detail="Invalid credentials")
    token_data = verify_token(token, credentials_exception)
    
    user = await collection_name.find_one(
        {"email": token_data.username}, 
        {"_id": 0, "password": 0}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Ensure profile_image exists
    if "profile_image" not in user or not user["profile_image"]:
        user["profile_image"] = DEFAULT_PROFILE_PIC

    return user

@router.get("/profile/image/{filename}")
async def get_profile_image(filename: str):
    """Serve profile images."""
    image_path = os.path.join(IMAGEDIR, filename)
    if not os.path.exists(image_path):
        image_path = DEFAULT_PROFILE_PIC
    return FileResponse(image_path)
