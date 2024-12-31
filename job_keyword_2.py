import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Function to recommend jobs based on keyword
def recommend_by_keyword(keyword, df, tfidf, tfidf_matrix, top_n=5):
    keyword_vector = tfidf.transform([keyword])
    similarity_scores = cosine_similarity(keyword_vector, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    return df.iloc[top_indices][['title', 'company', 'location', 'Employment type']]

# Streamlit interface
st.title("Job Data Explorer")

# Load and preprocess data
df = pd.read_csv("preprocessed_data.csv")
df = df.dropna(subset=["description"])

# TF-IDF Vectorizer
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df["description"])

# Main selection: Keyword search or visualizations
choice = st.radio("Hello There ! What would you like to do ?", ("Enter a keyword for job recommendations", "View data visualizations"))

if choice == "Enter a keyword for job recommendations":
    keyword = st.text_input("Enter a keyword or phrase to find relevant jobs:")

    if keyword:
        recommendations = recommend_by_keyword(keyword, df, tfidf, tfidf_matrix)
        st.write("Job Recommendations:")
        st.table(recommendations)

elif choice == "View data visualizations":
    st.write("## Data Visualizations")
    visualization = st.selectbox(
        "Select a visualization to display:",
        [
            "None",
            "Répartition des types d'emploi",
            "Distribution des années d'expérience",
            "Années d'expérience par niveau de seniorité",
            "Top 10 des localisations des emplois",
            "Top 10 des fonctions les plus fréquentes",
            "Répartition des années d'expérience par niveau de seniorité",
            "Percentage of the Most Repeated Companies",
        ],
    )

    # Display selected visualization
    if visualization == "Répartition des types d'emploi":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df, y='Employment type', order=df['Employment type'].value_counts().index, ax=ax)
        ax.set_title("Répartition des types d'emploi")
        ax.set_xlabel("Nombre d'emplois")
        st.pyplot(fig)

    elif visualization == "Distribution des années d'expérience":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['Years_experience'], bins=15, color='blue', ax=ax)
        ax.set_title("Distribution des années d'expérience")
        st.pyplot(fig)

    elif visualization == "Années d'expérience par niveau de seniorité":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='Years_experience', y='Seniority level', ax=ax)
        ax.set_title("Années d'expérience par niveau de seniorité")
        st.pyplot(fig)

    elif visualization == "Top 10 des localisations des emplois":
        fig, ax = plt.subplots(figsize=(10, 6))
        top_locations = df['location'].value_counts().head(10)
        sns.barplot(x=top_locations.values, y=top_locations.index, ax=ax)
        ax.set_title("Top 10 des localisations des emplois")
        ax.set_xlabel("Nombre d'emplois")
        ax.set_ylabel("Localisation")
        st.pyplot(fig)

    elif visualization == "Top 10 des fonctions les plus fréquentes":
        fig, ax = plt.subplots(figsize=(10, 6))
        top_functions = df['Job function'].value_counts().head(10)
        sns.barplot(x=top_functions.values, y=top_functions.index, palette='magma', ax=ax)
        ax.set_title("Top 10 des fonctions les plus fréquentes")
        ax.set_xlabel("Nombre d'emplois")
        ax.set_ylabel("Fonction")
        st.pyplot(fig)

    elif visualization == "Répartition des années d'expérience par niveau de seniorité":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(data=df, x='Seniority level', y='Years_experience', palette='muted', ax=ax)
        ax.set_title("Répartition des années d'expérience par niveau de seniorité")
        st.pyplot(fig)

    elif visualization == "Percentage of the Most Repeated Companies":
        top_20_companies = df['company'].value_counts().head(13)
        colors = plt.cm.Blues(np.linspace(0.3, 1, len(top_20_companies)))
        fig, ax = plt.subplots(figsize=(10, 7))
        top_20_companies.plot.pie(autopct='%1.1f%%', startangle=90, colors=colors, legend=False, ax=ax)
        ax.set_title('Percentage of the Most Repeated Companies')
        st.pyplot(fig)
