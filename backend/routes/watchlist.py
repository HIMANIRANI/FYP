from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from motor.motor_asyncio import AsyncIOMotorClient
from typing import List

from ..models.watchlist_model import WatchlistItem, AddWatchlistRequest, DeleteWatchlistRequest
from ..configurations.config import settings
from .jwttoken import verify_token
from ..models.token_model import TokenData

router = APIRouter()

# MongoDB setup
client = AsyncIOMotorClient(settings.MONGODB_URI)
db = client.userdata
collection = db["watchlist"]

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    return verify_token(token, credentials_exception)

@router.post("/watchlist/add", status_code=201)
async def add_to_watchlist(
    request: AddWatchlistRequest,
    token_data: TokenData = Depends(get_current_user)
):
    # Prevent duplicate for this user and scrip code
    existing = await collection.find_one({"user_id": token_data.username, "scrip.code": request.scrip.code})
    if existing:
        raise HTTPException(status_code=400, detail="Scrip already in watchlist")
    item = {
        "user_id": token_data.username,
        "scrip": request.scrip.dict()
    }
    await collection.insert_one(item)
    return {"message": "Added to watchlist"}

@router.get("/watchlist/get", response_model=List[WatchlistItem])
async def get_watchlist(token_data: TokenData = Depends(get_current_user)):
    items = await collection.find({"user_id": token_data.username}).to_list(length=100)
    return items

@router.delete("/watchlist/delete")
async def delete_from_watchlist(
    request: DeleteWatchlistRequest,
    token_data: TokenData = Depends(get_current_user)
):
    result = await collection.delete_many({
        "user_id": token_data.username,
        "scrip.code": {"$in": request.codes}
    })
    return {"deleted_count": result.deleted_count}
