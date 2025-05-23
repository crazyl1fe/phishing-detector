from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("phishing_rf_model.pkl")
scaler = joblib.load("scaler.pkl")

# Home page (with form)
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        result = "Phishing" if prediction == 1 else "Legitimate"
        return render_template('index.html', prediction_text=f"Result: {result}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
