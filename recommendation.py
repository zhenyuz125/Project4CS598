import pandas as pd
import numpy as np

# Load sMatrix and RMat
sMatrix = pd.read_csv('subset_sMatrix.csv', index_col=0)
# Define the myIBCF function
def myIBCF(newuser, R, S, top_n=10, popularity_ranking_file="movie_popularity.csv"):
    """
    Item-Based Collaborative Filtering (IBCF) for recommending movies to a new user.
    Args:
        newuser (pd.Series): A 3706-by-1 vector of ratings for the movies, with many entries as NA.
        R (pd.DataFrame): The original user-item rating matrix (users as rows, movies as columns).
        S (pd.DataFrame): The precomputed similarity matrix (movies as rows/columns).
        top_n (int): Number of top recommendations to return.
        popularity_ranking_file (str): Filepath to save/load movie popularity rankings.
    Returns:
        list: List of top N recommended movies.
    """
    # Step 1: Load or compute movie popularity ranking
    if not pd.io.common.file_exists(popularity_ranking_file):
        # Compute movie popularity as the sum of non-NA ratings
        movie_popularity = R.notna().sum().sort_values(ascending=False)
        movie_popularity.to_csv(popularity_ranking_file, header=False)
    else:
        # Load movie popularity from the precomputed file
        movie_popularity = pd.read_csv(popularity_ranking_file, index_col=0, header=None).squeeze("columns")
    # Step 2: Identify movies not rated by the new user
    unrated_movies = newuser[newuser.isna()].index
    # Step 3: Compute predictions for unrated movies
    predictions = {}
    for movie_i in unrated_movies:
        # Similar movies to `movie_i` and their similarities
        similar_movies = S.loc[movie_i].dropna()
        similar_movies = similar_movies[similar_movies.index.difference(unrated_movies)]  # Only use rated movies

        # Ratings by the new user for similar movies
        rated_movies = newuser[similar_movies.index].dropna()

        # Calculate the prediction using the weighted formula
        numerator = np.sum(similar_movies[rated_movies.index] * rated_movies)
        denominator = np.sum(similar_movies[rated_movies.index].abs())

        # Prediction is valid if denominator > 0
        predictions[movie_i] = numerator / denominator if denominator > 0 else np.nan

    # Step 4: Sort predictions and select top N
    predicted_ratings = pd.Series(predictions).sort_values(ascending=False).dropna()
    recommended_movies = predicted_ratings.index[:top_n].tolist()
    # Step 5: Fill remaining slots with popular movies
    if len(recommended_movies) < top_n:
        already_rated_or_recommended = newuser.dropna().index.union(recommended_movies)
        remaining_movies = movie_popularity.index.difference(already_rated_or_recommended)
        additional_movies = remaining_movies[:(top_n - len(recommended_movies))].tolist()
        recommended_movies.extend(additional_movies)

    return recommended_movies