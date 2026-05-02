import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
from datetime import datetime
import json

def save_model_and_parameters(grid_search):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"churn_logreg_best_{timestamp}.pkl"
    model_path = f"./src/models/{model_name}"

    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"✅ Model saved to: {model_path}")

    meta = {
    "model_name": model_name,
    "best_params": grid_search.best_params_,
    "best_cv_f1": round(grid_search.best_score_, 4),
    "saved_at": timestamp,
}
    meta_name = f"churn_logreg_meta_{timestamp}.json"
    meta_path = f"./src/models/{meta_name}"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=4)

    print(f"✅ Metadata saved to: {meta_path}")


def train():
    df = load_data(file_path=file_path)
    df.dropna(inplace=True)
    X, y, preprocessor = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    param_grid = {
    "classifier__C": [0.01, 0.1, 1, 10, 100],          # regularisation strength
    "classifier__penalty": ["l1", "l2"],                 # regularisation type
    "classifier__solver": ["liblinear", "saga"],         # solvers that support l1+l2
    "classifier__class_weight": [None, "balanced"],      # handles churn class imbalance
    }

    grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",           # F1 on positive class (Churn = 1)
    cv=5,                   # 5-fold stratified cross-validation
    n_jobs=-1,              # use all CPU cores
    verbose=0,              # print progress
    refit=True,             # refit best model on full training data
    return_train_score=True # lets you check for overfitting later
    )

    
    grid_search.fit(X_train, y_train)

    print("Best F1 Score (CV):", round(grid_search.best_score_, 4))
    print("Best Params:       ", grid_search.best_params_)


    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # Saving best model to models folder
    save_model_and_parameters(grid_search)

if __name__ == "__main__":
    train()