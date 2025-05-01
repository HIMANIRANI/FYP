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
    profile_image: str = "profile_pictures/default.jpg"  # Default image path

    class Config:
        """Configuration for the Pydantic model."""
        fields = {"confirmPassword": {"exclude": True}}
