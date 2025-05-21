from fastapi import APIRouter, HTTPException
from ..models.feedback_model import Feedback
from motor.motor_asyncio import AsyncIOMotorClient
from ..configurations.config import settings

router = APIRouter(prefix="/api/feedback")

client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata
feedback_collection = db["feedback"]

@router.post("/submit")
async def submit_feedback(feedback: Feedback):
    await feedback_collection.insert_one(feedback.dict())
    return {"message": "Feedback submitted successfully"}

@router.get("/all")
async def get_all_feedback():
    feedbacks = await feedback_collection.find().to_list(length=1000)
    return feedbacks 