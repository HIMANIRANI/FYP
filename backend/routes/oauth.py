from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jwttoken import verify_token

# Define the OAuth2PasswordBearer instance
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Dependency to get the current user based on the provided token
def get_current_user(token: str = Depends(oauth2_scheme)):
    # Define the exception to raise if token validation fails
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # Verify the provided token and return the user's information
    return verify_token(token, credentials_exception)
