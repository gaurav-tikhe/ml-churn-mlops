# Customer Churn Prediction — End-to-End MLOps Pipeline

> An end-to-end machine learning system for predicting customer churn, featuring a full data pipeline, experiment tracking with MLflow, a REST API built with FastAPI, and Docker-based deployment.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Key Insights from EDA](#key-insights-from-eda)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Docker Deployment](#docker-deployment)
- [API Usage](#api-usage)
- [Experiment Tracking](#experiment-tracking)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

This project demonstrates a production-grade MLOps workflow applied to a binary classification problem — predicting whether a telecom customer will churn. It covers every stage of the ML lifecycle: data ingestion and preprocessing, model training, experiment tracking, artifact management, and serving predictions via a containerised REST API.

---

## Problem Statement

Customer churn is a critical business problem in the telecom industry. Retaining an existing customer is significantly cheaper than acquiring a new one. This system predicts, given a customer's service and billing profile, whether they are likely to churn — enabling proactive retention strategies.

---

## Dataset

**Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

The dataset contains information about ~7,000 telecom customers including demographics, account details, services subscribed, and churn status.

---

## Key Insights from EDA

- **Class imbalance:** ~73% non-churners vs ~27% churners — addressed during model training.
- **Contract type is the strongest predictor:** Month-to-month customers churn at ~40%, compared to ~3% for two-year contracts.
- **Internet service type matters:** Fiber optic users churn at a significantly higher rate than DSL users.
- **Security services reduce churn:** Customers without `OnlineSecurity` churn far more frequently.
- **Tenure is negatively correlated with churn:** Churned customers average ~18 months tenure vs ~38 months for retained customers.
- **Pricing effect:** Churned customers pay higher average monthly charges but have lower total charges due to shorter tenure.

---

## System Architecture

```
Raw Data
   │
   ▼
Preprocessing & Feature Engineering
   │
   ▼
Model Training (Scikit-learn)
   │
   ▼
Experiment Tracking (MLflow)
   │
   ▼
Saved Model Artifact (joblib)
   │
   ▼
REST API (FastAPI)
   │
   ▼
Containerised Deployment (Docker)
```

---

## Project Structure

```
ml-churn-mlops/
├── data/                        # Raw and processed datasets
├── notebooks/                   # EDA and experimentation notebooks
├── src/
│   ├── api/
│   │   └── app.py               # FastAPI application
│   ├── pipeline/                # Data preprocessing pipeline
│   └── train.py                 # Model training script
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
├── mlflow.db                    # Local MLflow tracking store
└── README.md
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10 |
| ML Framework | Scikit-learn |
| Experiment Tracking | MLflow |
| API Framework | FastAPI + Uvicorn |
| Containerisation | Docker |
| Data Processing | Pandas, NumPy |
| Model Serialisation | Joblib |

---

## Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerised deployment)
- pip

### Local Setup

**1. Clone the repository:**

```bash
git clone https://github.com/gaurav-tikhe/ml-churn-mlops.git
cd ml-churn-mlops
```

**2. Create a virtual environment and install dependencies:**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**3. Train the model:**

```bash
python src/train.py
```

This runs the training pipeline, logs metrics and parameters to MLflow, and saves the model artifact.

**4. View experiment results in MLflow UI:**

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open [http://localhost:5000](http://localhost:5000) to explore runs, metrics, and artifacts.

**5. Run the API locally:**

```bash
uvicorn src.api.app:app --reload --port 8000
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

### Docker Deployment

**Build the image:**

```bash
docker build -t churn-api .
```

**Run the container:**

```bash
docker run -p 8000:8000 churn-api
```

The API will be available at [http://localhost:8000](http://localhost:8000).

---

## API Usage

### `POST /predict`

Submit a customer profile to receive a churn prediction.

**Request body (JSON):**

```json
{
  "tenure": 12,
  "MonthlyCharges": 70.5,
  "TotalCharges": 846.0,
  "Contract": "Month-to-month",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  ...
}
```

**Response:**

```json
{
  "churn_prediction": 1,
  "churn_probability": 0.82
}
```

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI).

---

## Experiment Tracking

All training runs are tracked in MLflow and stored in `mlflow.db`. Each run logs:

- **Parameters** — model hyperparameters, preprocessing choices
- **Metrics** — accuracy, ROC-AUC, F1-score, precision, recall
- **Artifacts** — serialised model file

To compare runs and select the best model, launch the MLflow UI as described in the setup steps above.

---

## Roadmap

- [ ] Model monitoring and data drift detection
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Cloud deployment (AWS / GCP / Azure)
- [ ] Hyperparameter tuning with Optuna
- [ ] Feature store integration

