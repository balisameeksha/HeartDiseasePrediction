# 💓 Heart Disease Prediction Web App

This project predicts the risk of heart disease using clinical input parameters and a machine learning model. It includes a **Flask web application** with an interactive HTML form.

## 🚀 Features

- Accepts 13 clinical features as input
- Predicts heart disease probability using a trained **XGBoost** model
- Real-time results shown via HTML frontend
- Clean and responsive UI
- API endpoint available for backend use

## 🛠 Technologies Used

- Python
- Flask
- Scikit-learn
- XGBoost
- SMOTE for data balancing
- HTML + CSS

## 📁 Project Structure

Heart_disease_pred/
├── app.py # Flask backend
├── model.pkl # Trained ML model
├── scaler.pkl # Scaler for input features
├── templates/
│ └── index.html # HTML user interface
└── README.md


## ⚙️ How to Run

1. Clone the repo:
git clone https://github.com/balisameeksha/HeartDiseasePrediction.git
cd HeartDiseasePrediction

2. Install required packages:
pip install -r requirements.txt

3. Run the app:
python app.py

4. Open in browser:
http://localhost:5000


