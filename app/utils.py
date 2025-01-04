import os
from pathlib import Path
import pickle

import requests
from dotenv import load_dotenv, find_dotenv
import streamlit as st

# загрузка переменных окружения из .env файла
load_dotenv(find_dotenv())

# TMDB API ключ
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status() # вызов HTTPError при неудаче
        data = response.json()
        poster_path = data.get('poster_path') # Используем .get(), чтобы избежать ошибки KeyError, если 'poster_path' отсутствует
        if poster_path:
            full_path = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return full_path
        else:
            st.warning(f"No poster found for movie ID: {movie_id}")
            return None
    except requests.exceptions.HTTPError as errh:
        st.error(f"HTTP Error: {errh}")
        st.stop()
    except requests.exceptions.ConnectionError as errc:
        st.error(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        st.error(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        st.error(f"Something went wrong: {err}")
    return None

def recommend(movie,movies,similarity):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []
    for i in distances[1:11]:  
        
        movie_id = movies.iloc[i[0]].id
        recommended_movie_posters.append(fetch_poster(movie_id))
        recommended_movie_names.append(movies.iloc[i[0]].title)

    return recommended_movie_names, recommended_movie_posters

def improved_recommendations(title,movies,similarity):
    index = movies[movies['title'] == title].index
    if len(index) == 0:
        st.warning(f"No movie found with the title '{title}'.")
        return None
    
    idx = index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    movies_qualified = movies.iloc[movie_indices][['id','title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies_qualified[movies_qualified['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies_qualified[movies_qualified['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)

    def weighted_rating(x, C=C, m=m):
        v = x['vote_count']
        R = x['vote_average']
        return (v / (v + m) * R) + (m / (m + v) * C)

    qualified_movies = movies_qualified[(movies_qualified['vote_count'] >= m) & (movies_qualified['vote_count'].notnull()) & (movies_qualified['vote_average'].notnull())]
    qualified_movies['vote_count'] = qualified_movies['vote_count'].astype('int')
    qualified_movies['vote_average'] = qualified_movies['vote_average'].astype('int')
    qualified_movies['wr'] = qualified_movies.apply(lambda x: weighted_rating(x), axis=1)
    qualified_movies = qualified_movies.sort_values('wr', ascending=False).head(10)

    return qualified_movies