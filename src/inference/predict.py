import joblib
import pandas as pd
from src.utils.logger import get_logger

MODEL_PATH = "src/models/churn_logreg_best_20260504_205659.pkl"
logger = get_logger(__name__)

def load_model():
    model = joblib.load(MODEL_PATH)
    return model

model = load_model()

def predict(input_data: dict):
    try:
        df = pd.DataFrame([input_data])
        logger.info(f"Input Dataframe : {df.to_dict()}")
        prediction = model.predict(df)
        logger.info(f"Model prediction: {prediction}")
        return prediction[0]
    except Exception as e:
        logger.error(f"Inference error : {e}")
        raise