from flask import Flask, request, jsonify

from utils import load_model, transform_data


app = Flask(__name__)

model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = transform_data([data['data']])
    predictions = model.predict(predict_request)
    
    return jsonify({'predictions': predictions.tolist()})


if __name__ == "__main__":
    app.run(port=8000)