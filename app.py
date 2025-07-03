from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import joblib
import os

app = Flask(__name__)

# Load the model and preprocessing components
with open('feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

scaler = joblib.load('scaler.joblib')
model = joblib.load('heart_disease_model.joblib')

def validate_input(data):
    required_features = set(feature_names)
    provided_features = set(data.keys())
    
    if not required_features.issubset(provided_features):
        missing = required_features - provided_features
        return False, f"Missing features: {missing}"
    
    # Validate data types
    try:
        for feature in feature_names:
            if feature == 'oldpeak':
                float(data[feature])
            else:
                int(data[feature])
    except ValueError as e:
        return False, f"Invalid value for feature {feature}. Must be numeric."
    
    return True, "Valid input"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        
        # Validate input
        is_valid, message = validate_input(data)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure correct feature order
        input_df = input_df[feature_names]
        
        # Scale features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        probability = model.predict_proba(scaled_features)[0][1]
        
        # Apply custom threshold for classification
        threshold = 0.4
        prediction = 1 if probability >= threshold else 0
        
        # Prepare response
        response = {
            "prediction": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "probability": float(probability),
            "prediction_code": int(prediction)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "API is running"}), 200

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    probability = None

    if request.method == 'POST':
        try:
            # Collect form data
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
                'thal': int(request.form['thal'])
            }

            # Make prediction
            scaled_features = scaler.transform(pd.DataFrame([data]))
            probability = model.predict_proba(scaled_features)[0][1]
            prediction_code = 1 if probability >= 0.4 else 0
            prediction = "Heart Disease" if prediction_code == 1 else "No Heart Disease"
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction, probability=probability)

if __name__ == '__main__':
    app.run(debug=True, port=5000)