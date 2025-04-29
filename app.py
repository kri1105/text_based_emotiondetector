from flask import Flask, request, render_template
import joblib
from train import clean_text
import os

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        text = request.form['text']
        cleaned_text = clean_text(text)
        emotion = model.predict([cleaned_text])[0]
        return render_template('result.html', text=text, emotion=emotion.capitalize())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)