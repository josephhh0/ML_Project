import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to preprocess and cluster jobs
def cluster_jobs(df, n_clusters=5):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df["description"])
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    df["cluster"] = clusters  # Add cluster labels to the dataframe
    
    return tfidf, tfidf_matrix, kmeans, df

# Function to recommend jobs based on cluster
def recommend_by_cluster(keyword, df, tfidf, tfidf_matrix, kmeans, top_n=5):
    keyword_vector = tfidf.transform([keyword])
    cluster_label = kmeans.predict(keyword_vector)[0]  # Predict the cluster for the keyword
    
    # Filter jobs within the same cluster
    cluster_jobs = df[df["cluster"] == cluster_label]
    
    # Calculate cosine similarity for jobs in the same cluster
    cluster_matrix = tfidf_matrix[df["cluster"] == cluster_label]
    similarity_scores = cosine_similarity(keyword_vector, cluster_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    
    return cluster_jobs.iloc[top_indices][['title', 'company', 'location', 'Employment type']]

# Streamlit interface
st.title("Job Keyword Analyzer with Clustering")

keyword = st.text_input("Please Enter a keyword or phrase to find relevant jobs:")

# Preprocess data once
df = pd.read_csv("preprocessed_data.csv")
df = df.dropna(subset=["description"])

# Cluster jobs and prepare data
tfidf, tfidf_matrix, kmeans, df = cluster_jobs(df)

if keyword:
    recommendations = recommend_by_cluster(keyword, df, tfidf, tfidf_matrix, kmeans)
    st.write("Job Recommendations:")
    st.table(recommendations)
