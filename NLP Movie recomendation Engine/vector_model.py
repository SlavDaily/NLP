import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np

#----------------------------------------------------------------------------------------------------------------
#This code is a vector model film search engine.
#It searches for 5 most similar movies to the specified one, based on word similarity
#between titles, description, genres and other text information.
#----------------------------------------------------------------------------------------------------------------

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('tmdb_5000_movies.csv')

# Replace NaN values with 'None'
df = df.replace(np.nan, 'None')

# Create a new column 'X' and concatenate relevant information from existing columns
df['X'] = ''
for index, row in df.iterrows():
    # Extract relevant information from columns
    data_string1 = row.title
    data_string2 = row.genres
    data_string3 = row.tagline
    data_string4 = row.overview
    data_string5 = row.production_companies
    data_string6 = row.production_countries
    data_string7 = row.keywords
    
    # Parse the JSON strings in certain columns
    data2 = json.loads(data_string2)
    data5 = json.loads(data_string5)
    data6 = json.loads(data_string6)
    data7 = json.loads(data_string7)

    # Extract names and concatenate them into a single string
    names_combined2 = ', '.join(item['name'] for item in data2)
    names_combined5 = ', '.join(item['name'] for item in data5)
    names_combined6 = ', '.join(item['name'] for item in data6)
    names_combined7 = ', '.join(item['name'] for item in data7)

    # Combine all relevant information into a single string
    data_tuple = (data_string1, names_combined2, data_string3, data_string4, names_combined5, names_combined6, names_combined7)
    data = ','.join(data_tuple)

    # Assign the concatenated string to the new 'X' column
    df.at[index, 'X'] = data

# Use TF-IDF Vectorizer to convert text data into numerical vectors
tfidf = TfidfVectorizer(max_features=3000)
X_train = tfidf.fit_transform(df.X)

# Take user input to choose a movie number
num = int(input("Choose movie number between 0 and 4802: "))

# Print details of the selected movie
print("")
print(df.iloc[num].title)
print(df.iloc[num].overview)
print('-------------------------------------------------------------------------------------------------------------------------')
print("")
print("MOST SIMILAR MOVIES ARE:")
print("")

# Use cosine similarity to find most similar movies
query = X_train[num]
cosine_similarities = cosine_similarity(X_train, query)
closest_indices = np.argsort(cosine_similarities[:, 0])[-6:-1][::-1]

# Print details of the most similar movies
for i in closest_indices:
    print(df.iloc[i].title)
    print(df.iloc[i].overview)
    print()
