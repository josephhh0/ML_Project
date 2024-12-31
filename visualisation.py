import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("preprocessed_data.csv")

plt.figure(figsize=(10, 6))
sns.countplot(data=data, y='Employment type', order=data['Employment type'].value_counts().index)
plt.title("Répartition des types d'emploi")
plt.xlabel("Nombre d'emplois")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data['Years_experience'], bins=15, color='blue')
plt.title("Distribution des années d'expérience")
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Years_experience', y='Seniority level')
plt.title("Années d'expérience par niveau de seniorité")
plt.xlabel("Années d'expérience")
plt.show()

plt.figure(figsize=(10, 6))
top_locations = data['location'].value_counts().head(10)
sns.barplot(x=top_locations.values, y=top_locations.index )
plt.title("Top 10 des localisations des emplois")
plt.xlabel("Nombre d'emplois")
plt.ylabel("Localisation")
plt.show()

plt.figure(figsize=(10, 6))
top_functions = data['Job function'].value_counts().head(10)
sns.barplot(x=top_functions.values, y=top_functions.index, palette='magma')
plt.title("Top 10 des fonctions les plus fréquentes")
plt.xlabel("Nombre d'emplois")
plt.ylabel("Fonction")
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(data=data, x='Seniority level', y='Years_experience', palette='muted')
plt.title("Répartition des années d'expérience par niveau de seniorité")
plt.show()

top_20_companies = data['company'].value_counts().head(13)
colors = plt.cm.Blues(np.linspace(0.3, 1, 20)) 

plt.figure(figsize=(10, 7))
top_20_companies.plot.pie(autopct='%1.1f%%', startangle=90, colors=colors, legend=False)
plt.title('Percentage of the Most Repeated Companies')
plt.show()

