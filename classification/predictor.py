import re
import joblib 
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#loading TD-IDF vectorizer and model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
nb_model = joblib.load("models/nb_model.pkl")
pac_model = joblib.load("models/pac_model.pkl")

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

    #naive bayes and PAC weigthed combination
    prob_nb = nb_model.predict_proba(vector)[:, 1]
    score_pac = pac_model.decision_function(vector)
    
    final_score = 0.65 * prob_nb + 0.35 * score_pac
    verdict = "FAKE" if final_score > 0.5 else "REAL"
    
    return verdict

