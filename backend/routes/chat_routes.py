from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from ..services.jwt_handler import decodeJWT
from datetime import datetime
from ..configurations.config import settings
from motor.motor_asyncio import AsyncIOMotorClient

from ..models.model import \
    PredictionPipeline  # Should work with this structure

router = APIRouter(prefix="/api/chat")

print("Router defined successfully")

# Load the AI Model once during startup
pipeline = PredictionPipeline()

# MongoDB setup for chat messages
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata
chat_collection = db["chat_messages"]

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        response = pipeline.make_predictions(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history")
async def get_chat_history(current_user: dict = Depends(decodeJWT)):
    user_id = current_user["user_id"]
    messages = await chat_collection.find({"user_id": user_id}).sort("timestamp", 1).to_list(100)
    return {"messages": [
        {
            "role": msg.get("role"),
            "content": msg.get("content"),
            "sender": msg.get("sender"),
            "receiver": msg.get("receiver"),
            "timestamp": msg.get("timestamp"),
            "message_limit": msg.get("message_limit", None)
        }
        for msg in messages
    ]}

async def store_message(user_id, sender, receiver, role, content, message_limit):
    await chat_collection.insert_one({
        "user_id": user_id,
        "sender": sender,
        "receiver": receiver,
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow(),
        "message_limit": message_limit
    })

if __name__ == "__main__":
    print("chat_routes.py executed directly")
