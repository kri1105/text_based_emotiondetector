# 🧠 Text-Based Emotion Detector

A simple **machine learning project** that detects emotions (joy, anger, sadness, fear, etc.) from text using **NLP** and **Support Vector Machines (SVM)**.

---

## 🚀 Overview

This project processes text data, cleans it, extracts features using **TF-IDF**, and trains a **LinearSVC** model to classify emotions.  
It can be integrated into chatbots, social media monitoring tools, or sentiment analysis systems.

---

## 📁 Project Structure

```
text_based_emotiondetector/
├── data/emotions.csv      # Dataset
├── train.py               # Training script
├── model.joblib           # Saved model
├── app.py                 # Flask web app
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<kri1105>/text_based_emotiondetector.git
cd text_based_emotiondetector
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
```

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python train.py
```

### 5. Run the Web Application

```bash
python app.py
```

---

## 📬 Usage

- Open your browser and go to: [http://127.0.0.1:5000/](http://127.0.0.1:5000/)
- Enter text to detect its emotion.

---

## 📝 License

MIT License

