"""
This module defines the User model using Pydantic.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """
    Represents a user with necessary fields for authentication.
    """

    firstName: str
    lastName: str
    email: EmailStr
    password: str
    confirmPassword: str = Field(exclude=True)
    is_premium: bool = False
    message_count: int = 0
    last_message_date: Optional[datetime] = None


class UserMessageLimit(BaseModel):
    """
    Represents a user's message limit status.
    """
    user_id: str
    message_count: int = 0
    is_premium: bool = False
    last_message_date: Optional[datetime] = None

    def can_send_message(self) -> bool:
        """
        Check if user can send a message based on their subscription status and message count.
        """
        if self.is_premium:
            return True
        return self.message_count < 2

    def increment_message_count(self):
        """
        Increment the message count for non-premium users.
        """
        if not self.is_premium:
            self.message_count += 1
            self.last_message_date = datetime.utcnow()
