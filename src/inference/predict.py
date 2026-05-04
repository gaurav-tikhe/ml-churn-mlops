import joblib
import pandas as pd

MODEL_PATH = "src/models/churn_logreg_best_20260504_205659.pkl"

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]