# 🧠 Text-Based Emotion Detector

A simple **machine learning project** that detects emotions (joy, anger, sadness, fear, etc.) from text using **NLP** and **Support Vector Machines (SVM)**.

---

## 🚀 Overview
This project processes text data, cleans it, extracts features using **TF-IDF**, and trains a **LinearSVC** model to classify emotions.  
It can be integrated into chatbots, social media monitoring tools, or sentiment analysis systems.

---

## 📁 Project Structure
text_based_emotiondetector/
├── data/emotions.csv # Dataset
├── train_model.py # Training script
├── model.joblib # Saved model
├── requirements.txt # Dependencies
└── README.md # Documentation

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/text_based_emotiondetector.git
cd text_based_emotiondetector

### 2. Create and Activate Virtual Environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

### 3. Install Dependencies
pip install -r requirements.txt

### 4. Train the Model


Then run:

python app.py

