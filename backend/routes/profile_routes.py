from fastapi import APIRouter, Depends, Form, HTTPException
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from ..configurations.config import settings
from ..services.auth_services import hash_password, verify_password, verify_token

router = APIRouter(prefix="/api/profile")

client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata
collection_name = db["users"]
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

@router.put("/update")
async def update_profile(
    firstName: Optional[str] = Form(None),
    lastName: Optional[str] = Form(None),
    oldPassword: Optional[str] = Form(None),
    newPassword: Optional[str] = Form(None),
    token: str = Depends(oauth2_scheme),
):
    credentials_exception = HTTPException(status_code=401, detail="Invalid credentials")
    token_data = verify_token(token, credentials_exception)
    user_email = token_data.username

    user = await collection_name.find_one({"email": user_email})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = {}
    if firstName is not None:
        update_data["firstName"] = firstName
    if lastName is not None:
        update_data["lastName"] = lastName
    if oldPassword and newPassword:
        if not verify_password(oldPassword, user["password"]):
            raise HTTPException(status_code=400, detail="Old password is incorrect")
        update_data["password"] = hash_password(newPassword)

    if update_data:
        result = await collection_name.update_one(
            {"email": user_email},
            {"$set": update_data}
        )
        if result.modified_count == 0:
            raise HTTPException(status_code=400, detail="No changes made to profile")
        updated_user = await collection_name.find_one(
            {"email": user_email},
            {"_id": 0, "password": 0}
        )
        return updated_user
    return user

@router.get("/get")
async def get_profile(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(status_code=401, detail="Invalid credentials")
    token_data = verify_token(token, credentials_exception)
    user = await collection_name.find_one(
        {"email": token_data.username},
        {"_id": 0, "password": 0}
    )
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
