from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..models.model import \
    PredictionPipeline  # Should work with this structure

router = APIRouter()

print("Router defined successfully")

# Load the AI Model once during startup
pipeline = PredictionPipeline()

class QueryRequest(BaseModel):
    question: str

@router.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        response = pipeline.make_predictions(request.question)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("chat_routes.py executed directly")
