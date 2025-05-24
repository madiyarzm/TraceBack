import re
import joblib 
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#loading TD-IDF vectorizer and model
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
nb_model = joblib.load("models/nb_model.pkl")
pac_model = joblib.load("models/pac_model.pkl")


