# MovieMind AI — Hybrid Movie Recommendation System

## Internship Project
This project implements the **Movie Recommendation System** listed in the internship project document.

**Objective:** Suggest movies based on user preferences using machine learning.

### Features
- Content-based filtering using TF-IDF and cosine similarity.
- Collaborative filtering using user-rating similarity.
- Hybrid recommendation score combining both approaches.
- Streamlit interactive UI.
- Genre-based fallback recommendation.
- Included demo MovieLens-style CSV dataset.
- Jupyter notebook for explanation and experimentation.

## Project structure
```text
Movie_Recommendation_System/
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
├── report.pdf
├── Movie_Recommendation_System.ipynb
└── data/
    ├── movies.csv
    └── ratings.csv
```

## Run locally
1. Open a terminal in this folder.
2. Install packages:
   `pip install -r requirements.txt`
3. Start the app:
   `streamlit run app.py`

## How it works
1. Movie genres are converted into TF-IDF vectors.
2. Cosine similarity finds movies with similar genre profiles.
3. User ratings are arranged into a user-item matrix.
4. Similar users contribute to collaborative scores.
5. Content and collaborative scores are normalized and combined.
6. The Streamlit UI displays the top 5 unseen movies.

## Using the full MovieLens dataset
The included CSVs are intentionally small so the project runs immediately. For a larger experiment, replace:
- `data/movies.csv`
- `data/ratings.csv`

with MovieLens-format files having the same column names.

## Interview explanation
The main ML concepts demonstrated are:
- TF-IDF
- cosine similarity
- collaborative filtering
- hybrid recommendation
- feature representation
- train/test-style experimentation
- Streamlit deployment

## Note
Recommendations are for demonstration/learning purposes and should not be treated as a guarantee of a user's preferences.
