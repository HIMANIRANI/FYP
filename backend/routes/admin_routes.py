from fastapi import APIRouter
from motor.motor_asyncio import AsyncIOMotorClient
from ..configurations.config import settings

router = APIRouter(prefix="/api/admin")

client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata
users_collection = db["users"]

@router.get("/users")
async def get_all_users():
    users = await users_collection.find({}, {"_id": 0, "email": 1, "firstName": 1, "lastName": 1, "is_premium": 1}).to_list(length=1000)
    return users 