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

        mlflow server --backend-store-uri sqlite:///config/mlflow.db

2) start app.py

        python app.py

3) test the result
      
        curl -X POST -H 'Content-Type: application/json' -d '{"data": [3,"Graduate","No",5000000,12700000,14,865,4700000,8100000,19500000,6300000]}' http://127.0.0.1:8080/predict
