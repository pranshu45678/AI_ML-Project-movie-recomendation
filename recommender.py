"""
Hybrid Movie Recommendation System
- Content-based filtering: TF-IDF over genres
- Collaborative filtering: user-item rating matrix + cosine similarity
- Hybrid score: weighted combination of both
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent / "data"

def load_data():
    movies = pd.read_csv(DATA_DIR / "movies.csv")
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    return movies, ratings

def build_content_model(movies):
    text = movies["genres"].fillna("").str.replace("|", " ", regex=False)
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(text)
    similarity = cosine_similarity(matrix)
    return vectorizer, similarity

def build_collaborative_model(movies, ratings):
    matrix = ratings.pivot_table(index="userId", columns="movieId",
                                  values="rating", fill_value=0)
    sim_users = cosine_similarity(matrix)
    return matrix, sim_users

def content_recommendations(movie_id, movies, similarity, n=5):
    if movie_id not in set(movies.movieId):
        return pd.DataFrame()
    idx = movies.index[movies.movieId == movie_id][0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    indices = [i for i, _ in scores if movies.iloc[i].movieId != movie_id][:n]
    result = movies.iloc[indices].copy()
    result["score"] = [similarity[idx, i] for i in indices]
    return result

def collaborative_recommendations(user_id, movies, ratings, n=5):
    matrix = ratings.pivot_table(index="userId", columns="movieId",
                                 values="rating", fill_value=0)
    if user_id not in matrix.index:
        return pd.DataFrame()

    user_vector = matrix.loc[user_id].values.reshape(1, -1)
    sims = cosine_similarity(user_vector, matrix.values)[0]
    sims[list(matrix.index).index(user_id)] = 0

    weighted = sims @ matrix.values
    denom = np.abs(sims).sum() + 1e-9
    predicted = weighted / denom

    rated = set(ratings.loc[ratings.userId == user_id, "movieId"])
    candidates = []
    for movie_id, score in zip(matrix.columns, predicted):
        if movie_id not in rated:
            candidates.append((movie_id, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    ids = [x[0] for x in candidates[:n]]
    scores = [x[1] for x in candidates[:n]]
    result = movies[movies.movieId.isin(ids)].copy()
    score_map = dict(candidates)
    result["score"] = result["movieId"].map(score_map)
    return result.sort_values("score", ascending=False)

def hybrid_recommendations(user_id, favorite_movie_id, movies, ratings,
                           content_similarity, n=5, content_weight=0.6):
    """
    Combine normalized content and collaborative scores.
    If user has no collaborative signal, content still works.
    """
    content = content_recommendations(
        favorite_movie_id, movies, content_similarity, n=len(movies)
    )
    content_scores = dict(zip(content.movieId, content.score))

    matrix = ratings.pivot_table(index="userId", columns="movieId",
                                  values="rating", fill_value=0)
    collab_scores = {}
    if user_id in matrix.index:
        user_vector = matrix.loc[user_id].values.reshape(1, -1)
        sims = cosine_similarity(user_vector, matrix.values)[0]
        sims[list(matrix.index).index(user_id)] = 0
        predicted = (sims @ matrix.values) / (np.abs(sims).sum() + 1e-9)
        collab_scores = dict(zip(matrix.columns, predicted))

    rated = set(ratings.loc[ratings.userId == user_id, "movieId"])
    max_c = max(content_scores.values(), default=1) or 1
    max_r = max(collab_scores.values(), default=1) or 1

    rows = []
    for movie_id in movies.movieId:
        if movie_id == favorite_movie_id or movie_id in rated:
            continue
        c = content_scores.get(movie_id, 0) / max_c
        r = collab_scores.get(movie_id, 0) / max_r if collab_scores else 0
        score = content_weight*c + (1-content_weight)*r
        rows.append((movie_id, c, r, score))

    rows.sort(key=lambda x: x[3], reverse=True)
    result = movies[movies.movieId.isin([x[0] for x in rows[:n]])].copy()
    details = {x[0]: (x[1], x[2], x[3]) for x in rows}
    result["content_score"] = result.movieId.map(lambda x: details[x][0])
    result["collab_score"] = result.movieId.map(lambda x: details[x][1])
    result["hybrid_score"] = result.movieId.map(lambda x: details[x][2])
    return result.sort_values("hybrid_score", ascending=False)

def recommend_by_genres(user_genres, movies, n=5):
    """
    Simple content-based preference search for users who do not select a known movie.
    """
    wanted = set(g.strip().lower() for g in user_genres)
    def score(genres):
        gs = set(g.lower() for g in genres.split("|"))
        return len(gs & wanted) / max(len(wanted), 1)
    result = movies.copy()
    result["score"] = result["genres"].apply(score)
    return result.sort_values(["score", "title"], ascending=[False, True]).head(n)
