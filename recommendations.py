import numpy as np
import pandas as pd
import re
from datetime import datetime
import joblib
import requests
from sklearn.neighbors import NearestNeighbors
import json


def clean_dataest(dataset):
    movies_dataset = dataset
    current_date = datetime.now()
    movies_dataset['release_date'] = pd.to_datetime(movies_dataset['release_date'], errors='coerce')
    movies_dataset = movies_dataset[movies_dataset['release_date'] <= current_date]
    movies_dataset = movies_dataset.dropna(subset=['overview'])
    movies_dataset = movies_dataset.dropna(subset=['keywords'])
    movies_dataset = movies_dataset.dropna(subset=['genres'])
    movies_dataset = movies_dataset[movies_dataset['keywords'].apply(lambda keywords: 'softcore' not in keywords)]
    movies_dataset = movies_dataset.drop_duplicates(subset=['title', 'release_date'], keep='first')
    # movies_dataset.to_csv('/Users/peeyush/github repos/movieSearch/movie_dataset.csv')
    return movies_dataset


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]"," ",text)
    return text


def get_prompt_embeddings(user_prompt):
    
  API_TOKEN = "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"
  API_URL = "https://api-inference.huggingface.co/models/WhereIsAI/UAE-Large-V1"
  headers = {"Authorization": f"Bearer {API_TOKEN}"}

  query = {"inputs": clean_text(user_prompt)}

  response = requests.post(API_URL, headers=headers, json=query)

  prompt_embedding = response.json()

  prompt_embedding = np.array([prompt[0] for prompt in prompt_embedding]).reshape(1,-1)
  return prompt_embedding


def get_recommended_movies(embeddings, sorting=False):
  distances, indices = knn.kneighbors(embeddings)

  recommended_movies = []
  recommended_movies_overview = []
  recommended_movies_keywords = []
  for i in indices.flatten():
      recommended_movies.append(movies_dataset.iloc[i].to_dict())

  if(sorting):
    return sorted(recommended_movies, key=lambda x: x['release_date'], reverse=True)
  else:
    return recommended_movies


def print_recommended_movies(recommended_movies):
  for movie in recommended_movies:
    print(f"{movie['title']} ({movie['release_date'].year})")
    
    
def search_movies_by_title(query):
    matching_movies = movies_dataset[movies_dataset['title'].str.contains(query, case=False)]
    if not matching_movies.empty:
        return matching_movies
    else:
        print(f"No movies found with title containing '{query}'.")
        return None
    
    
def get_movie_embeddings_by_title(title, embeddings):

    movie_row = movies_dataset[movies_dataset['title'].str.lower() == title.lower()]

    if not movie_row.empty:
        movie_index = movie_row.index[0]
        movie_embedding = embeddings[movie_index]
        return movie_embedding
    else:
        print(f"Movie with title '{title}' not found.")
        return None
    
def get_watched_movies_embeddings(movie_titles, embeddings):
    sum_embeddings = np.zeros_like(embeddings[0])
    for title in movie_titles:
        movie_embedding = get_movie_embeddings_by_title(title, embeddings)
        if movie_embedding is not None:
            sum_embeddings += movie_embedding
    return np.array(sum_embeddings)


def movies_by_search(prompt):
    embeddings = None
    if prompt in prompt_embeddings:
        embeddings = prompt_embeddings[prompt]
    else:
        embeddings = get_prompt_embeddings(prompt)
        prompt_embeddings[prompt] = embeddings
        update_prompt_embeddings()
    recommended_movies = get_recommended_movies(embeddings)
    return recommended_movies

def movies_by_movie(title):
    movie_embeddings = get_movie_embeddings_by_title(title)
    recommended_movies = get_recommended_movies(movie_embeddings)
    return recommended_movies


def load_dataset():
    movies_dataset = pd.read_csv("/Users/peeyush/github repos/movieSearch/movie_dataset.csv")
    movies_dataset = clean_dataest(movies_dataset)
    
    return movies_dataset

def load_movies_embeddings():
    movies_embeddings =joblib.load("/Users/peeyush/github repos/movieSearch/movie_context_embeddings_uae.npy")
    movies_embeddings = np.array([movie[0] for movie in movies_embeddings])
    return movies_embeddings

def load_knn_model():
    return joblib.load("/Users/peeyush/github repos/movieSearch/knn_model.joblib")

def load_prompt_embeddings():
    prompt_embeddings_serializable = None
    file_path = "/Users/peeyush/github repos/movieSearch/prompt_embeddings.json"
    try:  
        with open(file_path, 'r') as file:
            prompt_embeddings_serializable = json.load(file)
    except:
        pass
    
    if prompt_embeddings_serializable and len(prompt_embeddings_serializable) > 0:
        prompt_embeddings = {key: np.array(value) for key, value in prompt_embeddings_serializable.items()}
    else:
        prompt_embeddings = {}
    
    return prompt_embeddings
    
def update_prompt_embeddings():
    file_path = "/Users/peeyush/github repos/movieSearch/prompt_embeddings.json"
    prompt_embeddings_serializable = {key: value.tolist() for key, value in prompt_embeddings.items()}

    with open(file_path, 'w') as file:
       json.dump(prompt_embeddings_serializable, file)

movies_dataset = load_dataset()
movies_embeddings = load_movies_embeddings()        
knn =  load_knn_model()
prompt_embeddings = load_prompt_embeddings()

