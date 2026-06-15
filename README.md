# 🏥 Healthcare Innovation - AI Drug Recommendation System

An AI-powered healthcare application that recommends suitable medicines based on user symptoms using Machine Learning. The system leverages a Support Vector Machine (SVM) model trained on medical symptom-drug datasets to provide intelligent drug recommendations through an interactive web interface.

---

## 🚀 Features

- 🤖 AI-based Drug Recommendation
- 🩺 Symptom-Based Medicine Prediction
- 📊 Machine Learning Model using SVM
- 🌐 Interactive Web Interface
- 📁 CSV-based Medical Dataset
- ⚡ Fast and Accurate Predictions
- 🔄 Automated Data Preprocessing
- 📈 Model Training and Evaluation

---

## 🛠️ Tech Stack

### Frontend
- HTML5
- Tailwind CSS
- JavaScript

### Backend
- Python
- Flask

### Machine Learning
- Scikit-Learn
- Support Vector Machine (SVM)
- MultiLabelBinarizer

### Data Processing
- Pandas
- NumPy

### Model Storage
- Joblib

---

## 📂 Project Structure

```
Healthcare-Innovation/
│
├── app.py
├── train_model.py
├── drug_data.csv
├── index.html
│
├── drug_recommendation_model.pkl
├── feature_columns.pkl
├── mlb_encoder.pkl
│
└── README.md
```

---

## 🔍 How It Works

1. User enters symptoms.
2. Symptoms are preprocessed and converted into machine-readable features.
3. The trained SVM model analyzes the symptoms.
4. The system predicts the most suitable medicine.
5. Results are displayed through a modern web interface.

---

## 🧠 Machine Learning Workflow

### Data Preprocessing
- Cleans symptom data
- Handles missing values
- Converts symptoms into feature vectors

### Feature Engineering
- MultiLabelBinarizer converts symptoms into one-hot encoded features

### Model Training
- Support Vector Machine (SVM) Classifier
- Stratified train-test split
- Accuracy evaluation

### Model Deployment
- Saved using Joblib
- Loaded into Flask application for real-time predictions

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/Healthcare-Innovation.git
```

### Navigate to Project

```bash
cd Healthcare-Innovation
```

### Install Dependencies

```bash
pip install flask pandas numpy scikit-learn joblib
```

### Train Model

```bash
python train_model.py
```

### Run Application

```bash
python app.py
```

### Open Browser

```text
http://127.0.0.1:5000
```

---

## 📊 Dataset

The project uses a healthcare dataset containing:

- Drug Names
- Symptoms
- Medical Conditions
- Treatment Information

The dataset is stored in:

```text
drug_data.csv
```

---

## 🎯 Applications

- Healthcare Assistance
- Medical Decision Support
- Drug Recommendation Systems
- Symptom Analysis
- AI-Powered Healthcare Solutions

---

## 🔮 Future Enhancements

- Disease Prediction Module
- Doctor Recommendation System
- Patient History Tracking
- Chatbot Integration
- Voice-Based Symptom Input
- Mobile Application
- Deep Learning Models
- Cloud Deployment

---

## 👩‍💻 Author

**Trupthi Niranjan**

Information Science Engineering Student

---

## 📜 License

This project is developed for educational, research, and learning purposes.
