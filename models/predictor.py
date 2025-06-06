import re
import joblib 
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Get the absolute path to the models directory
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    #loading TD-IDF vectorizer and model with absolute paths
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    nb_model = joblib.load(os.path.join(MODELS_DIR, "nb_model.pkl"))
    pac_model = joblib.load(os.path.join(MODELS_DIR, "pac_model.pkl"))
    print("Models loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")
    raise

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return ' '.join(words)

def predict_verdict(text):
    cleaned = preprocess(text)
    vector = vectorizer.transform([cleaned])

    #naive bayes and PAC weighted combination
    prob_nb = nb_model.predict_proba(vector)[:, 1]
    score_pac = pac_model.decision_function(vector)
    
    # Print debug information
    print(f"Input text: {text}")
    print(f"Cleaned text: {cleaned}")
    print(f"Naive Bayes probability: {prob_nb[0]:.4f}")
    print(f"PAC score: {score_pac[0]:.4f}")
    
    #final_score = 0.65 * prob_nb + 0.35 * score_pac
    final_score = prob_nb
    print(f"Final score: {final_score[0]:.4f}")
    
    verdict = "FAKE" if final_score > 0.5 else "REAL"
    print(f"Predicted verdict: {verdict}")
    
    return verdict

