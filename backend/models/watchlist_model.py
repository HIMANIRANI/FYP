from pydantic import BaseModel, Field
from typing import Optional, Dict

class PriceInfo(BaseModel):
    open: float
    close: float
    diff: float
    prevClose: float

class ScripInfo(BaseModel):
    code: str
    name: str
    price: PriceInfo
    tradedShares: int
    amount: float

class WatchlistItem(BaseModel):
    user_id: Optional[str] = None  # Will be set from JWT
    scrip: ScripInfo

class AddWatchlistRequest(BaseModel):
    scrip: ScripInfo

class DeleteWatchlistRequest(BaseModel):
    codes: list[str]  # List of scrip codes to delete

