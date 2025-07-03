from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)

# === Load Model and Scaler ===
try:
    model = joblib.load("heart_disease_model.joblib")
except FileNotFoundError:
    model = joblib.load("model.pkl")  # fallback

try:
    scaler = joblib.load("scaler.joblib")
except FileNotFoundError:
    scaler = joblib.load("scaler.pkl")  # fallback

# Define the feature names in correct order (13 features)
feature_names = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak',
    'slope', 'ca', 'thal'
]

# === Helper Function ===
def validate_input(data):
    required = set(feature_names)
    received = set(data.keys())
    if not required.issubset(received):
        missing = required - received
        return False, f"Missing features: {missing}"

    try:
        for feature in feature_names:
            if feature == 'oldpeak':
                float(data[feature])
            else:
                int(data[feature])
    except ValueError:
        return False, f"Invalid data type for {feature}. Must be numeric."
    return True, "Valid"

# === API Endpoint ===
@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.json
    is_valid, msg = validate_input(data)
    if not is_valid:
        return jsonify({"error": msg}), 400

    input_df = pd.DataFrame([data])[feature_names]
    scaled = scaler.transform(input_df)
    prob = model.predict_proba(scaled)[0][1]
    pred = "Heart Disease" if prob >= 0.4 else "No Heart Disease"

    return jsonify({
        "prediction": pred,
        "probability": round(prob, 4),
        "prediction_code": int(prob >= 0.4)
    })

# === Health Check ===
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "running", "message": "App is healthy"}), 200

# === HTML Form View ===
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            data = {
                'age': int(request.form['age']),
                'sex': int(request.form['sex']),
                'cp': int(request.form['cp']),
                'trestbps': int(request.form['trestbps']),
                'chol': int(request.form['chol']),
                'fbs': int(request.form['fbs']),
                'restecg': int(request.form['restecg']),
                'thalach': int(request.form['thalach']),
                'exang': int(request.form['exang']),
                'oldpeak': float(request.form['oldpeak']),
                'slope': int(request.form['slope']),
                'ca': int(request.form['ca']),
                'thal': int(request.form['thal']),
            }

            input_df = pd.DataFrame([data])[feature_names]
            scaled = scaler.transform(input_df)
            prob = model.predict_proba(scaled)[0][1]
            prediction = "Heart Disease" if prob >= 0.4 else "No Heart Disease"
            probability = f"{prob * 100:.2f}%"

        except Exception as e:
            prediction = f"Error: {str(e)}"
            probability = "N/A"

    return render_template("index.html", prediction=prediction, probability=probability)

# === Run the App ===
if __name__ == '__main__':
    app.run(debug=True, port=5000)
