from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from .auth_services import verify_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def decodeJWT(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_token(token, credentials_exception)
    return {"user_id": token_data.username} 