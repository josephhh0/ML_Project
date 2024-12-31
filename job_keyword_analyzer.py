import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def recommend_by_keyword(keyword, df, tfidf, tfidf_matrix, top_n=5):
    keyword_vector = tfidf.transform([keyword])
    similarity_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    return df.iloc[top_indices][['title']]

# Streamlit interface
st.title("Job Keyword Analyzer")

keyword = st.text_input("Enter a keyword or phrase to find relevant jobs:")


# Preprocess data once
df = pd.read_csv("preprocessed_data.csv")
df = df.dropna(subset=["description"])

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["description"])

if keyword:
    recommendations = recommend_by_keyword(keyword, df, tfidf, tfidf_matrix)
    st.write("Job Recommendations:")
    st.table(recommendations)
