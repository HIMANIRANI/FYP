import base64
import json
import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import (Depends, FastAPI, HTTPException, Request, Response,
                     WebSocket, WebSocketDisconnect, status)
from fastapi.concurrency import run_in_threadpool  # Added for threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Adjust path to import PredictionPipeline
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from backend.models.model import PredictionPipeline
from backend.routes.auth_routes import router as auth_router
from backend.routes.payments import PaymentRequest, initiate_payment

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# FastAPI setup
app = FastAPI(
    title="NEPSE-Navigator",
    description="A system for finance",
    version="1.0.0",
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Register auth routes
app.include_router(auth_router)

# # Model initialization
# pipeline = PredictionPipeline()
# try:
#     logger.info("Loading PredictionPipeline components...")
#     pipeline.load_model_and_tokenizers()
#     pipeline.load_sentence_transformer()
#     pipeline.load_reranking_model()
#     pipeline.load_embeddings()
#     logger.info("PredictionPipeline loaded successfully.")
# except Exception as e:
#     logger.error(f"Failed to load PredictionPipeline: {str(e)}", exc_info=True)
#     raise

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Welcome to the NEPSE-Navigator System!",
        "documentation_url": "/docs",
        "authentication_routes": "/auth",
        "payment_routes": "/api/initiate-payment",
        "websocket_endpoint": "/ws"
    }

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("✅ WebSocket connection established")
#     try:
#         while True:
#             data = await websocket.receive_text()
#             try:
#                 query = json.loads(data).get("question", "")
#                 if not query:
#                     await websocket.send_text(json.dumps({"response": "❗ Please provide a question."}))
#                     continue
#                 logger.info(f"📥 Received query: {query}")
#                 try:
#                     response = pipeline.make_predictions(query)
#                     logger.info("Sending response")
#                     await websocket.send_text(json.dumps({"response": response}))
#                     logger.info("📤 Sent response.")
#                 except Exception as e:
#                     logger.error(f"Prediction error: {e}", exc_info=True)
#                     await websocket.send_text(json.dumps({"response": f"An error occurred: {e}"}))
#             except json.JSONDecodeError as e:
#                 logger.warning("Invalid JSON received")
#                 await websocket.send_text(json.dumps({"response": f"Invalid JSON: {e}"}))
#     except WebSocketDisconnect as e:
#         logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason or 'none'}")
#     except Exception as e:
#         logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
#     finally:
#         logger.info("WebSocket connection closed")
        

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("✅ WebSocket connection established")
    try:
        while True:
            data = await websocket.receive_text()
            try:
                query = json.loads(data).get("question", "")
                if not query:
                    await websocket.send_text(json.dumps({"response": "❗ Please provide a question."}))
                    continue

                logger.info(f"📥 Received query: {query}")

                # 🔍 Simple keyword match for fundamental analysis
                if "fundamental analysis of nabil bank" in query.lower():
                    response = nabil_fundamental_analysis()
                    await websocket.send_text(json.dumps({"response": response}))
                    continue

                try:
                    response = pipeline.make_predictions(query)
                    logger.info("Sending response")
                    await websocket.send_text(json.dumps({"response": response}))
                    logger.info("📤 Sent response.")
                except Exception as e:
                    logger.error(f"Prediction error: {e}", exc_info=True)
                    await websocket.send_text(json.dumps({"response": f"An error occurred: {e}"}))
            except json.JSONDecodeError as e:
                logger.warning("Invalid JSON received")
                await websocket.send_text(json.dumps({"response": f"Invalid JSON: {e}"}))
    except WebSocketDisconnect as e:
        logger.info(f"WebSocket disconnected: code={e.code}, reason={e.reason or 'none'}")
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("WebSocket connection closed")

@app.post("/api/initiate-payment")
async def payment_endpoint(request: PaymentRequest):
    try:
        logger.info(f"Payment request: {request.dict()}")
        response = await initiate_payment(request)
        logger.info("Payment initiated.")
        return response
    except Exception as e:
        logger.error(f"Payment initiation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Payment failed: {str(e)}")

@app.get("/success")
async def handle_payment_success(request: Request):
    method = request.query_params.get("method")
    data = request.query_params.get("data")
    if not method or not data:
        logger.warning("Missing success parameters")
        raise HTTPException(status_code=400, detail="Missing method or data parameter")

    if method == "esewa":
        try:
            decoded_data = base64.b64decode(data).decode("utf-8")
            payment_data = json.loads(decoded_data)
            if payment_data.get("status") == "COMPLETE":
                return RedirectResponse(url=f"http://localhost:5173/success?method={method}&data={data}")
            raise HTTPException(status_code=400, detail="Payment not completed")
        except Exception as e:
            logger.error(f"Payment decode error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error processing payment: {str(e)}")
    raise HTTPException(status_code=400, detail="Invalid payment method")

@app.get("/failure")
async def handle_payment_failure():
    logger.info("Payment failed, redirecting...")
    return RedirectResponse(url="http://localhost:5173/failure")

# Add this new route for stock data
@app.get("/api/stocks/today")
async def get_today_stocks():
    try:
        # Construct the path to today.json
        json_path = Path("backend/data/initial data/date/today.json")
        
        # If the file doesn't exist at the relative path, try absolute path
        if not json_path.exists():
            json_path = Path(os.path.dirname(os.path.abspath(__file__))) / "data" / "initial data" / "date" / "today.json"
        
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Stock data file not found")
            
        # Read and parse the JSON file
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Error parsing stock data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)