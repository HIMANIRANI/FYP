from typing import Optional

from pydantic import BaseModel


class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    exp: Optional[int] = None

class UserInfo(BaseModel):
    username: str
    exp: int

class LoginRequest(BaseModel):
    email: str
    password: str
