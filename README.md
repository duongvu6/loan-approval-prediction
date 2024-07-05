# Loan Approval Predict Project

- Problem Statement:
  - A loan approval dataset is a collection of financial records and associated information used to determine eligibility for loans. It includes various factors such as civic score, income, employment status, loan term, loan amount, asset value and loan status.

- Data Columns

  - Loan_id: the Number of Loan
  - no_of_dependents
  - education
  - self_employed
  - income_annum
  - loan_amount
  - loan_term
  - cibil_score
  - residential_assets_value
  - commercial_assets_value
  - luxury_assets_value
  - bank_asset_value
  - loan_status: Our target Column for prediction

## Start ML Project

1) start mlflow server (ui)

        mlflow server --backend-store-uri sqlite:///mlflow.db

2) start app.py

        python app.py
