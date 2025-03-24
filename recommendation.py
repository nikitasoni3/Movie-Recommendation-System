import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split, cross_validate
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from dotenv import load_dotenv
import requests
import time
import os

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_IMAGE_URL = "https://image.tmdb.org/t/p/w500"
session = requests.Session()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
movies_df = pd.read_csv(os.path.join(BASE_DIR, "Dataset/movie_cleaned_dataset.csv"))
ratings = pd.read_csv(os.path.join(BASE_DIR, "Dataset/ratings_cleaned_dataset.csv"))

movies_df['genres'] = movies_df['genres'].fillna("").apply(lambda g: g.split(", ") if isinstance(g, str) else g)
movies_df['tags'] = movies_df['tags'].fillna("")

def fetch_movie_poster(movie_id):
    try:
        if not movie_id:
            raise ValueError("movie_id is missing!")
        time.sleep(0.5)
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        response = session.get(url, timeout=5)
        if response.status_code != 200:
            raise ValueError(f"TMDb API error: {response.status_code}, Response: {response.text}")
        data = response.json()
        poster_path = data.get("poster_path")
        return f"{TMDB_IMAGE_URL}{poster_path}" if poster_path else None
    except Exception:
        return None

def format_recommendations(movie_list):
    formatted = []
    for movie in movie_list:
        movie_id = movie.get("movie_id")
        if movie_id is None:
            continue
        formatted.append({
            "title": movie.get("title", "Unknown Title"),
            "poster": fetch_movie_poster(movie_id) if movie_id else None
        })
    return formatted

def get_movie_list():
    return movies_df["title"].tolist()

def get_user_list():
    return ratings["userId"].unique().tolist()

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies_df['tags']).toarray()
similarity = cosine_similarity(vectors)

def recommend(movie, num_recommendations=3):
    movie_index = movies_df[movies_df['title'].str.lower() == movie.lower()].index
    if movie_index.empty:
        return [{"error": f"Movie '{movie}' not found in the database"}]
    movie_index = movie_index[0]
    distance = similarity[movie_index]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:num_recommendations + 1]
    recommended_movies = [
        {"title": movies_df.iloc[i[0]]["title"], "movie_id": movies_df.iloc[i[0]]["movie_id"]}
        for i in movies_list
    ]
    return format_recommendations(recommended_movies)

def recommend_by_genre(genre, num_recommendations=5):
    genre = genre.lower().strip()
    if isinstance(movies_df['genres'].iloc[0], str):
        movies_df['genres'] = movies_df['genres'].apply(lambda g: g.split(", ") if isinstance(g, str) else g)
    filtered_movies = movies_df[movies_df['genres'].apply(lambda g_list:
        isinstance(g_list, list) and any(genre in g.lower() for g in g_list))]
    return format_recommendations([
        {"title": row.title, "movie_id": row.movie_id}
        for _, row in filtered_movies.head(num_recommendations).iterrows()
    ])

def recommend_by_keywords(keyword, num_recommendations=5):
    keyword = keyword.lower().strip()
    filtered_movies = movies_df[movies_df['tags'].fillna("").str.contains(keyword, case=False, na=False)]
    return format_recommendations([
        {"title": row.title, "movie_id": row.movie_id}
        for _, row in filtered_movies.head(num_recommendations).iterrows()
    ])

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
model = SVD()
cross_validate(model, data, cv=5)
trainset, testset = train_test_split(data, test_size=0.2)
model.fit(trainset)

def recommend_movies(user_id, num_recommendations=3):
    movie_ids = movies_df['movie_id'].unique()
    watched_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    recommendations = sorted(
        [(movie_id, model.predict(user_id, movie_id).est) for movie_id in movie_ids if movie_id not in watched_movies],
        key=lambda x: x[1], reverse=True
    )[:num_recommendations]
    return format_recommendations([
        {
            "title": movies_df[movies_df['movie_id'] == movie[0]]['title'].values[0],
            "movie_id": movie[0]
        }
        for movie in recommendations
    ])

def hybrid_recommendation(movie_title, user_id, num_recommendations=5):
    content_recs = recommend(movie_title)
    collab_recs = recommend_movies(user_id, num_recommendations)
    unique_titles = set()
    merged_recs = []
    for rec in content_recs + collab_recs:
        title = rec["title"]
        if title not in unique_titles:
            unique_titles.add(title)
            merged_recs.append(rec)
    sorted_hybrid_recs = sorted(merged_recs, key=lambda x: x["title"])
    return sorted_hybrid_recs[:num_recommendations]
