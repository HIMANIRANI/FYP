from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from pymongo.collection import Collection

from ..configurations.config import settings


# Helper for ObjectId
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

# Pydantic models
class PriceModel(BaseModel):
    open: float
    close: float
    prevClose: float
    diff: float

class WatchlistItemIn(BaseModel):
    user_id: str
    scrip: str
    company_name: str
    price: PriceModel
    numTrans: int
    tradedShares: int
    amount: float

class WatchlistItemOut(WatchlistItemIn):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    added_at: datetime

    class Config:
        json_encoders = {ObjectId: str}

# Dependency to get MongoDB collection
def get_watchlist_collection() -> Collection:
    client = AsyncIOMotorClient(settings.MONGODB_URI)
    db = client.userdata  # Database for user-related data
    collection_name = db["users"]  # Collection to store user information

router = APIRouter()

@router.get("/watchlist", response_model=List[WatchlistItemOut])
def get_watchlist(user_id: str, collection: Collection = Depends(get_watchlist_collection)):
    items = list(collection.find({"user_id": user_id}))
    return items

@router.post("/watchlist/add", response_model=WatchlistItemOut)
def add_watchlist_item(item: WatchlistItemIn, collection: Collection = Depends(get_watchlist_collection)):
    # Prevent duplicate
    if collection.find_one({"user_id": item.user_id, "scrip": item.scrip}):
        raise HTTPException(status_code=400, detail="Scrip already in watchlist")
    doc = item.dict()
    doc["added_at"] = datetime.utcnow()
    result = collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc

@router.delete("/watchlist/{item_id}")
def delete_watchlist_item(item_id: str, collection: Collection = Depends(get_watchlist_collection)):
    result = collection.delete_one({"_id": ObjectId(item_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"detail": "Deleted"}
