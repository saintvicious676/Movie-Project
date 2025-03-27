# 🎬 Movie Recommendation System  

## Project Overview

This is the final deliverable for Group 3's Project 4: A machine learning-based movie recommendation system built using the MovieLens dataset and TMDB API. The goal of this project was to explore user interaction data, build predictive models, and deploy a web-based tool that returns personalized movie recommendations.

The application integrates multiple components across data engineering, model development, visualization, and deployment. We implemented collaborative filtering techniques alongside user behavior analysis to provide a responsive and data-informed recommendation experience.

---

## Key Components

### Data Engineering and ETL

We extracted and cleaned movie ratings, metadata, and user tags from the MovieLens and TMDB datasets. The cleaned data was structured and loaded into a PostgreSQL database to support efficient model training and web querying. Dask was used to process larger datasets in parallel.

- `data_cleaning.ipynb` – Data wrangling and preparation
- `movies.csv`, `tags_cleaned.csv`, `filtered_ratings.zip` – Clean datasets
- `MoviesSql.sql`, `speedsql.sql` – SQL queries for PostgreSQL integration

---

### Recommendation System Development

The core model utilizes matrix factorization through the Surprise library (SVD), trained on the user-item ratings matrix. We explored additional techniques including clustering (K-Means) and deep learning-based alternatives for potential enhancements.

Modeling steps included:
- Training/testing splits using cross-validation
- Parameter tuning for latent factors and regularization
- Confusion matrix and RMSE used for evaluation

Key files:
- `ml_model.html` – Technical overview of our modeling process
- `ml_app.html` – Front-end app interface for recommendations

---

### Data Visualization and Insights

To support interpretability and analysis, we created interactive plots using Plotly and Matplotlib. These visualizations show rating patterns by genre, user engagement behavior, and trends over time.

Sample insights include:
- Genre popularity and average ratings over time
- Heatmaps of user rating intensity
- Top movies by tag (e.g. “funny”)

Visualization files (can also be seen on our website):
- `average_rating_by_genre.html`
- `user_behavior_heatmap.html`
- `genre_popularity_sunburst.html`
- `top_movies_by_tag.html`

---

### Web Deployment

The system is deployed as a Flask web application that connects the trained model with a user interface. Dropdowns and sliders support real-time filtering, and Redis is used for caching to reduce load time and optimize performance. We also evaluated cloud storage options with AWS S3 for future scalability.

Deployment files:
- `finalindex.html` – Main landing page
- `deployment.html` – Hosting and system architecture
- `caching.html` – Caching strategies and performance review

---

## Team Contributions

| Name          | Responsibility                             |
|---------------|---------------------------------------------|
| Brandon D.    | Data Engineering, ETL, PostgreSQL           |
| Jake P.       | Model Development, Evaluation               |
| Tavneet       | Deep Learning Exploration, Model Tuning     |
| Kirsten F.    | Visualization, Insight Analysis             |
| Zilan Y.      | Flask Development, Redis, AWS Integration   |

---

## Datasets

- [MovieLens 25M Dataset](https://grouplens.org/datasets/movielens/)
- [TMDB API](https://www.themoviedb.org/documentation/api)

The MovieLens dataset includes over 32 million ratings and 2 million user-generated tags across 87,000+ movies. User IDs are anonymized and filtered to ensure data quality.
