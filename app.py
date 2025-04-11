from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
import nltk
import string
import re

nltk.download('punkt')

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('emotions.csv')

# Basic Preprocessing (Unicode + punctuation removal)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token.strip() != '']
    return ' '.join(tokens)

# Clean the data
data['clean_text'] = data['Sentences'].apply(preprocess_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['clean_text'])

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Label'])

# Train Classifier
classifier = LinearSVC()
classifier.fit(X, y)

# Evaluate
y_pred = classifier.predict(X)
accuracy = accuracy_score(y, y_pred)
classification_rep = classification_report(y, y_pred, target_names=label_encoder.classes_)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    clean_input = preprocess_text(text)
    vectorized_text = vectorizer.transform([clean_input])
    prediction = classifier.predict(vectorized_text)
    predicted_label = label_encoder.inverse_transform(prediction)[0]
    return render_template('index.html', text=text, predicted_label=predicted_label,
                           accuracy=round(accuracy * 100, 2), classification_rep=classification_rep)

if __name__ == '__main__':
    app.run(debug=True)
