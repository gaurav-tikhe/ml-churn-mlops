from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import pandas as pd

def preprocess_data(df:pd.DataFrame):
    X = df.drop("Churn",axis=1)
    y = df["Churn"].map({"Yes":1, "No":0})

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",numeric_transformer,numeric_features),
            ("cat",categorical_transformer,categorical_features)
        ]
    )
    return X,y,preprocessor