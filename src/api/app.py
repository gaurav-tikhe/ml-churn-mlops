from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict

app = FastAPI()

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
    return {"message": "Churn Prediction is running"}

@app.post("/predict")
def predict_churn(request: ChurnRequest):
    data = request.model_dump()
    prediction = predict(data)
    return {"prediction": int(prediction)}