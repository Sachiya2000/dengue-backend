from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)  # frontend එකෙන් requests allow කරන්න

# Model එක load කරගන්න
model = joblib.load('dengue_predictor.pkl')

# Test endpoint


@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'Backend is working!'})

# Prediction endpoint


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Input data DataFrame එකක් විදිහට හදන්න
        input_data = pd.DataFrame({
            'temperature_2m_mean': [data['temperature']],
            'precipitation_sum': [data['precipitation']],
            'windspeed_10m_max': [data['windspeed']]
        })

        # Prediction කරන්න
        prediction = model.predict(input_data)[0]

        return jsonify({
            'prediction': round(prediction, 2),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        })


if __name__ == '__main__':
    app.run(debug=True)
