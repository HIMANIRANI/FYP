import base64
import hashlib
import hmac
import logging
import os
import uuid

import httpx
from dotenv import load_dotenv
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic model for incoming payment data
class PaymentRequest(BaseModel):
    amount: str
    product_name: str
    transaction_id: str
    method: str

# Validate environment variables
def validate_env_vars():
    required_vars = ["BASE_URL", "ESEWA_MERCHANT_CODE", "ESEWA_SECRET_KEY", "KHALTI_SECRET_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise Exception(f"Missing environment variable: {var}")

# Generate eSewa signature
def generate_esewa_signature(secret_key: str, message: str) -> str:
    hmac_obj = hmac.new(secret_key.encode(), message.encode(), hashlib.sha256)
    return base64.b64encode(hmac_obj.digest()).decode()

# Payment initiation function
async def initiate_payment(request: PaymentRequest):
    try:
        validate_env_vars()
        logger.info(f"Received payment request: {request.dict()}")
        amount = request.amount
        product_name = request.product_name
        transaction_id = request.transaction_id
        method = request.method

        if not all([amount, product_name, transaction_id, method]):
            raise HTTPException(status_code=400, detail="Missing required fields")

        if method == "esewa":
            transaction_uuid = f"{uuid.uuid4()}-{int(uuid.uuid1().time)}"
            esewa_config = {
                "amount": amount,
                "tax_amount": "0",
                "total_amount": amount,
                "transaction_uuid": transaction_uuid,
                "product_code": os.getenv("ESEWA_MERCHANT_CODE"),
                "product_service_charge": "0",
                "product_delivery_charge": "0",
                "success_url": f"{os.getenv('BASE_URL')}/success?method=esewa",
                "failure_url": "http://localhost:5173/failure",
                "signed_field_names": "total_amount,transaction_uuid,product_code",
            }
            signature_string = f"total_amount={esewa_config['total_amount']},transaction_uuid={esewa_config['transaction_uuid']},product_code={esewa_config['product_code']}"
            signature = generate_esewa_signature(os.getenv("ESEWA_SECRET_KEY"), signature_string)

            logger.info(f"eSewa config: {esewa_config}, signature: {signature}")
            return JSONResponse(content={
                "amount": amount,
                "esewa_config": {**esewa_config, "signature": signature}
            })

        elif method == "khalti":
            khalti_config = {
                "return_url": f"{os.getenv('BASE_URL')}/success?method=khalti",
                "failure_url": "http://localhost:5173/failure",
                "website_url": os.getenv("BASE_URL"),
                "amount": int(float(amount) * 100),  # Convert to paisa
                "purchase_order_id": transaction_id,
                "purchase_order_name": product_name,
                "customer_info": {
                    "name": "Test User",
                    "email": "test@example.com",
                    "phone": "9800000000"
                }
            }
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://a.khalti.com/api/v2/epayment/initiate/",
                    headers={
                        "Authorization": f"Key {os.getenv('KHALTI_SECRET_KEY')}",
                        "Content-Type": "application/json"
                    },
                    json=khalti_config
                )
                if not response.is_success:
                    logger.error(f"Khalti API error: {response.text}")
                    raise HTTPException(status_code=500, detail="Khalti payment initiation failed")
                khalti_response = response.json()
                logger.info(f"Khalti response: {khalti_response}")
                return JSONResponse(content={"khalti_payment_url": khalti_response["payment_url"]})

        else:
            raise HTTPException(status_code=400, detail="Invalid payment method")

    except Exception as e:
        logger.error(f"Payment initiation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
