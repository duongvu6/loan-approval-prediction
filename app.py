import subprocess
import tqdm
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

from load_model import setup, load_model
from preprocess_data import preprocess_data


app = Flask(__name__)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = np.array(data['data'])
    predict_request = [4250,4,'Graduate','No',1100000,4000000,14,887,2400000,1500000,4200000,1600000]
    input_df = np.array(preprocess_data(pd.DataFrame([predict_request], columns=[
            "loan_id", "no_of_dependents", "education", "self_employed", "income_annum", 
            "loan_amount", "loan_term", "cibil_score", "residential_assets_value", 
            "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
        ])), dtype=np.int64)

    predictions = model.predict(input_df)
    
    return jsonify({'predictions': predictions.tolist()})


if __name__ == "__main__":
    # setup()
    app.run(debug=True, port=8080)