import streamlit as st
import pandas as pd
from recommender import (
    load_data, build_content_model, hybrid_recommendations,
    recommend_by_genres
)

st.set_page_config(page_title="MovieMind AI", page_icon="🎬", layout="wide")

movies, ratings = load_data()
_, content_similarity = build_content_model(movies)

st.title("🎬 MovieMind AI")
st.caption("Hybrid movie recommendation system using content-based and collaborative filtering.")

st.sidebar.header("Choose your preferences")
mode = st.sidebar.radio("Recommendation mode", ["Based on a movie", "Based on genres"])

if mode == "Based on a movie":
    user_id = st.sidebar.selectbox("Demo user", sorted(ratings.userId.unique()), index=0)
    favorite = st.sidebar.selectbox("Favorite movie", movies.title.tolist(), index=1)
    favorite_id = int(movies.loc[movies.title == favorite, "movieId"].iloc[0])
    weight = st.sidebar.slider("Content importance", 0.0, 1.0, 0.6, 0.1)

    if st.button("✨ Recommend Movies"):
        result = hybrid_recommendations(
            user_id, favorite_id, movies, ratings,
            content_similarity, n=5, content_weight=weight
        )
        st.subheader(f"Top recommendations for you")
        for _, row in result.iterrows():
            st.markdown(f"### 🎥 {row['title']}")
            st.write(f"Genres: {row['genres'].replace('|', ', ')}")
            st.progress(float(max(0, min(1, row["hybrid_score"]))))
            st.caption(
                f"Hybrid score: {row['hybrid_score']:.3f} | "
                f"Content: {row['content_score']:.3f} | "
                f"Collaborative: {row['collab_score']:.3f}"
            )
            st.divider()
else:
    options = ["Action", "Adventure", "Animation", "Children", "Comedy",
               "Crime", "Drama", "Fantasy", "Horror", "Romance", "Sci-Fi",
               "Thriller"]
    selected = st.multiselect("Select genres you like", options, default=["Action", "Sci-Fi"])
    if st.button("✨ Find Movies"):
        if not selected:
            st.warning("Please select at least one genre.")
        else:
            result = recommend_by_genres(selected, movies, n=5)
            st.subheader("Top 5 genre-based recommendations")
            for _, row in result.iterrows():
                st.markdown(f"### 🎥 {row['title']}")
                st.write(f"Genres: {row['genres'].replace('|', ', ')}")
                st.caption(f"Genre match score: {row['score']:.2f}")

st.markdown("---")
st.info("This internship demo uses a small MovieLens-style dataset included in the project. Replace the CSV files with the full MovieLens dataset for a larger experiment.")
