# Full updated Streamlit app code with dual recommendations (genre-based and user-based)

import pandas as pd
import dask.dataframe as dd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import streamlit as st
import re
# Dark mode aesthetic
st.markdown("""
<style>
h1 {
    color: #e50914;
    font-size: 2.8rem;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# Load and train SVD model for user-based collaborative filtering
@st.cache_resource
def train_svd(df):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    svd = SVD()
    svd.fit(trainset)
    return svd, trainset

# Recommend based on similar users' ratings
def recommend_by_user_similarity(
    movie_title, df, df_movies, svd_model, trainset,
    n=5, min_reviews=10
):
    matched = df_movies[df_movies['title'].str.lower() == movie_title.lower()]
    if matched.empty:
        return "Movie not found."

    movie_id = matched.iloc[0]['movieId']
    users_who_rated = df[df['movieId'] == movie_id]
    top_users = users_who_rated.sort_values(by='rating', ascending=False).head(50)['userId'].tolist()

    movie_scores = {}
    for uid in top_users:
        for movie in df['movieId'].unique():
            if movie != movie_id:
                pred = svd_model.predict(uid, movie)
                movie_scores[movie] = movie_scores.get(movie, []) + [pred.est]

    averaged_scores = {
        mid: sum(scores)/len(scores)
        for mid, scores in movie_scores.items() if len(scores) >= 3
    }

    top_movie_ids = sorted(averaged_scores, key=averaged_scores.get, reverse=True)[:n]
    result = df_movies[df_movies['movieId'].isin(top_movie_ids)].copy()

    # **New**: Filter out movies below `min_reviews`
    result = result[result['numRatings'] >= min_reviews]

    # Assign predicted rating
    result['Predicted Rating'] = result['movieId'].apply(
        lambda x: round(averaged_scores.get(x, 0), 2)
    )

    result = result.dropna(subset=['imdbId'])
    result['IMDb Link'] = result['imdbId'].apply(
        lambda x: f'<a href="https://www.imdb.com/title/tt{int(x):07d}/" target="_blank">IMDb</a>'
    )

    return result[['title', 'genres', 'year', 'Predicted Rating', 'IMDb Link']] \
                 .rename(columns={
                     'title': 'Movie Title',
                     'genres': 'Genres',
                     'year': 'Year'
                 })


# Load and cache data
@st.cache_data
def load_data():
    ddf = dd.read_csv("../../ml-32m/ratings.csv", encoding='ISO-8859-1', dtype={"userId": "object", "movieId": "object", "rating": "object"})
    dd_movies = dd.read_csv("Resources/movies.csv")
    dd_links = dd.read_csv("Resources/links.csv")
    df = ddf[['userId', 'movieId', 'rating']].compute()
    df = df[pd.to_numeric(df['userId'], errors='coerce').notnull() & pd.to_numeric(df['movieId'], errors='coerce').notnull() & pd.to_numeric(df['rating'], errors='coerce').notnull()]
    df['userId'] = df['userId'].astype(int)
    df['movieId'] = df['movieId'].astype(int)
    df['rating'] = df['rating'].astype(float)
    df_movies = dd_movies[['movieId', 'title', 'genres']].compute()
    df_links = dd_links[['movieId', 'imdbId']].compute()

    avg_ratings = df.groupby("movieId")["rating"].agg(avgRating="mean", numRatings="count").reset_index()
    avg_ratings["avgRating"] = avg_ratings["avgRating"].round(2)

    df_movies = df_movies.merge(avg_ratings, on="movieId", how="left")
    df_movies = df_movies.merge(df_links, on="movieId", how="left")
    df_movies['year'] = df_movies['title'].apply(lambda x: int(re.search(r'\((\d{4})\)', x).group(1)) if re.search(r'\((\d{4})\)', x) else None)
    df_movies['year'] = df_movies['year'].astype('Int64')
    return df, df_movies

df, df_movies = load_data()
svd_model, trainset = train_svd(df)

st.title("🍿 Fancy Another Movie?")

movie_input = st.selectbox("Enter a Movie Title You Like:", sorted(df_movies['title'].dropna().unique().tolist()))

def recommend_similar_movies(
    movie_title,
    df_movies,
    include_genres=None,
    exclude_genres=None,
    year_range=None,
    n=5,
    min_reviews=10
):
    matched = df_movies[df_movies['title'].str.lower() == movie_title.lower()]
    if matched.empty:
        return "Movie not found."

    input_genres = matched['genres'].iloc[0].split('|') if not matched.empty else []
    movie_id = matched['movieId'].iloc[0]

    genre_pattern = '|'.join(input_genres)
    recommendations = df_movies[df_movies['genres'].str.contains(genre_pattern, case=False, na=False)].copy()

    if include_genres:
        include_pattern = '|'.join(include_genres)
        recommendations = recommendations[recommendations['genres'].str.contains(include_pattern, case=False, na=False)]
    if exclude_genres:
        for g in exclude_genres:
            recommendations = recommendations[~recommendations['genres'].str.contains(g, case=False, na=False)]

    if year_range:
        recommendations = recommendations[
            (recommendations['year'] >= year_range[0]) &
            (recommendations['year'] <= year_range[1])
        ]

    # Exclude the original movie
    recommendations = recommendations[recommendations['movieId'] != movie_id]

    # **New filter**: Only keep movies with at least `min_reviews` ratings
    recommendations = recommendations[recommendations['numRatings'] >= min_reviews]

    recommendations = recommendations.dropna(subset=['imdbId'])
    recommendations['IMDb Link'] = recommendations['imdbId'].apply(
        lambda x: f'<a href="https://www.imdb.com/title/tt{int(x):07d}/" target="_blank">IMDb</a>'
    )
    recommendations['MovieLens Rating'] = recommendations.apply(
        lambda row: f"{row['avgRating']} ({int(row['numRatings']) if pd.notnull(row['numRatings']) else 0} reviews)",
        axis=1
    )

    def genre_overlap(genres):
        movie_genres = set(genres.split('|'))
        return len(set(input_genres) & movie_genres)

    recommendations['score'] = recommendations['genres'].apply(genre_overlap)

    return recommendations.sort_values(by=["score", "avgRating", "year"],
                                       ascending=[False, False, False]) \
                          .head(n)[
                              ['title', 'genres', 'year', 'MovieLens Rating', 'IMDb Link']
                          ].rename(columns={
                              'title': 'Movie Title',
                              'genres': 'Genres',
                              'year': 'Year'
                          })



all_genres = set()
df_movies['genres'].dropna().apply(lambda g: all_genres.update(g.split('|')))
all_genres = sorted(all_genres)

# Sidebar inputs
top_n_input = st.slider("Number of Recommendations:", min_value=1, max_value=20, value=5)
matched_row = df_movies[df_movies['title'].str.lower() == movie_input.lower()]
genres = matched_row['genres'].iloc[0].split('|') if not matched_row.empty else []
auto_tags = matched_row['genres'].iloc[0].split('|') if not matched_row.empty else []
include_genres = st.multiselect("Include Genre(s):", options=genres, default=auto_tags)
exclude_genres = st.multiselect("Exclude Genre(s):", options=all_genres, default=[])
min_reviews_input = st.number_input(
    "Minimum number of reviews:",
    min_value=1,
    max_value=100000,
    value=10,
    step=1
)
rating_range = st.slider("MovieLens Rating Range:", min_value=0.0, max_value=5.0, value=(0.0, 5.0), step=0.1)
year_min, year_max = int(df_movies['year'].min()), int(df_movies['year'].max())
year_range = st.slider("Year Range:", min_value=year_min, max_value=year_max, value=(1874, 2023), step=1, format="%d")

if st.button("Recommend Movies") and movie_input:
    with st.spinner("Finding similar movies..."):
        genre_recommendations = recommend_similar_movies(
            movie_input,
            df_movies,
            include_genres=include_genres,
            exclude_genres=exclude_genres,
            year_range=year_range,
            n=top_n_input,
            min_reviews=min_reviews_input
        )

        # Filter by rating range
        genre_recommendations = genre_recommendations[
            genre_recommendations['MovieLens Rating']
            .str.extract(r'(\d+\.\d+)')[0]
            .astype(float)
            .between(rating_range[0], rating_range[1])
        ]

        user_recommendations = recommend_by_user_similarity(
            movie_input,
            df,
            df_movies,
            svd_model,
            trainset,
            n=top_n_input,
            min_reviews=min_reviews_input
        )

        # ... same display logic as before ...


        table_style = """
        <style>
            table { width: 100%; table-layout: fixed; }
            th, td {
                word-wrap: break-word;
                text-align: left;
                vertical-align: top;
                white-space: normal;
            }
            th {
                white-space: nowrap;
                font-size: 15px;
                padding: 8px;
                text-align: left;
            }
            th:nth-child(1), td:nth-child(1) { width: 28%; }
            th:nth-child(2), td:nth-child(2) { width: 24%; }
            th:nth-child(3), td:nth-child(3) { width: 12%; }
            th:nth-child(4), td:nth-child(4) { width: 24%; }
            th:nth-child(5), td:nth-child(5) { width: 18%; }
        </style>
        """

        if isinstance(genre_recommendations, str):
            st.warning(genre_recommendations)
        else:
            st.subheader(f"🎯 Top {top_n_input} Movies Similar to '{movie_input}'")
            st.markdown(table_style, unsafe_allow_html=True)
            st.write(genre_recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)

        if isinstance(user_recommendations, str):
            st.warning(user_recommendations)
        else:
            st.subheader("🎬 Similar Users Also Watched…")
            st.markdown(table_style, unsafe_allow_html=True)
            st.write(user_recommendations.to_html(escape=False, index=False), unsafe_allow_html=True)

st.markdown("---")
st.caption("Powered by Dask | Built with Streamlit + IMDb")
