from pydantic import BaseModel, EmailStr

class Feedback(BaseModel):
    email: EmailStr
    message: str 