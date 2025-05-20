# Heart Disease Prediction with Real Target, SMOTE, and Model Comparison

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import warnings
import pickle
import joblib
warnings.filterwarnings("ignore")

# Step 2: Load dataset (make sure this dataset has a real 'target' column)
df = pd.read_csv("heart.csv")  # Loading the dataset

# Step 3: Data exploration
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Preprocessing
df.columns = df.columns.str.strip().str.lower()  # Clean column names
df['oldpeak'] = df['oldpeak'].astype(float)

# Step 5: Feature and target separation
X = df.drop('target', axis=1)
y = df['target']

# Step 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

# Step 8: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Step 9: Define models to compare
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Step 10: Train and evaluate each model
for name, model in models.items():
    print(f"\n===== {name} =====")
    model.fit(X_train_scaled, y_train_resampled)
    y_pred = model.predict(X_test_scaled)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 11: Visualize prediction for one sample
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_train_scaled, y_train_resampled)
sample = X_test_scaled[0:1]
probs = final_model.predict_proba(sample)[0]

plt.bar(['No Disease', 'Disease'], probs, color=['green', 'red'])
plt.title("Prediction Probability for a Sample")
plt.ylabel("Probability")
plt.show()

# Step 12: Save the model and preprocessing components
print("\nSaving model and components...")
# Save feature names
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Save the scaler
joblib.dump(scaler, 'scaler.joblib')

# Save the model
joblib.dump(final_model, 'heart_disease_model.joblib')

# Create a prediction function
def predict_heart_disease(data_dict):
    """
    Make heart disease predictions on new data.
    
    Args:
        data_dict (dict): Dictionary containing patient data with the following keys:
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal'
    
    Returns:
        tuple: (prediction (0 or 1), probability of heart disease)
    """
    # Load saved components
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('heart_disease_model.joblib')
    
    # Convert input dictionary to dataframe
    input_df = pd.DataFrame([data_dict])
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in input_df.columns:
            raise ValueError(f"Missing feature: {feature}")
    
    # Arrange features in the correct order
    input_df = input_df[feature_names]
    
    # Scale the features
    scaled_features = scaler.transform(input_df)
    
    # Make prediction
    probability = model.predict_proba(scaled_features)[0][1]
    
    # Adjust the threshold to 0.4
    threshold = 0.4
    prediction = 1 if probability >= threshold else 0
    
    # Print the probability explicitly for debugging
    print(f"Probability of Heart Disease: {probability:.2f}")
    
    return prediction, probability

# Example usage of the prediction function
example_patient = {
    'age': 60,
    'sex': 1,
    'cp': 2,
    'trestbps': 140,
    'chol': 185,
    'fbs': 0,
    'restecg': 0,
    'thalach': 155,
    'exang': 0,
    'oldpeak': 3,
    'slope': 0,
    'ca': 0,
    'thal': 1
}

print("\nTesting prediction function with example patient:")
prediction, probability = predict_heart_disease(example_patient)
print(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
print(f"Probability of Heart Disease: {probability:.2f}")