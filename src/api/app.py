from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict
from src.utils.logger import get_logger

app = FastAPI()
logger = get_logger(__name__)

# Define Input Schema
class ChurnRequest(BaseModel):
    customerID:str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def root():
    logger.info("Root Endpoint called")
    return {"message": "Churn Prediction is running"}

@app.post("/predict")
def predict_churn(request: ChurnRequest):
    try:
        data = request.model_dump()
        logger.info(f"Received request: {data}")

        prediction = predict(data)
        logger.info(f"Prediction result: {prediction}")
        return {"prediction": int(prediction)}
    
    except Exception as e:
        logger.error(f"Error during prediction :- {e}")
        return {"error": "Prediction failed"}