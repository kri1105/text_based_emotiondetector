import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
import re
import string

nltk.download('wordnet')
nltk.download('omw-1.4')

# Text preprocessing
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Load and prepare data
data = pd.read_csv("data/emotions.csv")
data['text'] = data['text'].apply(clean_text)
texts = data["text"]
labels = data["emotion"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Create and train model
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('clf', LinearSVC(C=1.0, class_weight='balanced'))
])

model.fit(X_train, y_train)

# Evaluate
print("Test accuracy:", model.score(X_test, y_test))

# Save model
joblib.dump(model, "model.joblib")
print("Model trained and saved!")