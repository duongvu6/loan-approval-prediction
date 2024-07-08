import subprocess
import tqdm
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

from utils import setup, load_model, transform_data


app = Flask(__name__)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = transform_data(data['data'])
    predictions = model.predict(predict_request)
    
    return jsonify({'predictions': predictions.tolist()})


if __name__ == "__main__":
    # setup()
    app.run(debug=True, port=8080)