# Phishing Detection Using Machine Learning

A simple web application that detects phishing websites using a trained Random Forest machine learning model. Built with Flask and scikit-learn.

## 🧠 Model Overview
- Dataset: [UCI Phishing Websites Dataset](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
- Algorithms: Random Forest (best performing model)
- Features: URL-based, domain-based, HTML/JS characteristics
- Accuracy: ~96%

## 📁 Project Structure

```
phishing-detection-app/
├── phishing_detection_app.py
├── train_model.py
├── phishing.csv
├── phishing_rf_model.pkl
├── scaler.pkl
├── templates/
│   └── index.html
└── README.md
```

## 🚀 Getting Started

### 1. Install dependencies
```bash
pip install flask scikit-learn pandas numpy joblib
```

### 2. Download dataset
Download and save as `phishing.csv`:
[Download](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/phishing.csv)

### 3. Train the model
```bash
python train_model.py
```

### 4. Run the app
```bash
python phishing_detection_app.py
```

Visit `http://127.0.0.1:5000/` to use the interface.
