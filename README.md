ML Churn Prediction with MLOps

Problem Statement :- 
Predict whether a customer will churn based on historical data.

Goal :-
Build an end-to-end ML system with:
  - Data pipeline
  - Model training
  - Experiment tracking
  - API deployment

Dataset used :- 
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Key Insights from EDA :-
1. There are ~73% non-churners and ~27% churners, indicating some class imbalance.
2. Month-to-month contracts churn at ~40% vs ~3% for two-year contracts
3. Fiber optic internet users churn more than DSL users
4. Customers without OnlineSecurity churn far more
5. Churned customers have much lower tenure (avg ~18 months vs ~38 for retained) (Also indicated by negative correlation between Tenure and Churn feature)
6. Churned customers pay higher monthly charges on average
7. TotalCharges is lower for churned (short tenure × high monthly)
