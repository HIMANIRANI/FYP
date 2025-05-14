"""
This module defines the User model using Pydantic.
"""

from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    """
    Represents a user with necessary fields for authentication.
    """

    firstName: str
    lastName: str
    email: EmailStr
    password: str
    confirmPassword: str = Field(exclude=True)

    class Config:
        """Configuration for the Pydantic model."""
        fields = {"confirmPassword": {"exclude": True}}
