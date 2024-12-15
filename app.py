import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL
import pandas as pd
import numpy as np
import random

from recommendation import sMatrix, myIBCF

# load the data, from prof
ratings = pd.read_csv('ratings.dat', sep='::', engine='python', header=None)
ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
movies = pd.read_csv('movies.dat', sep='::', engine='python', encoding="ISO-8859-1", header=None)
movies.columns = ['MovieID', 'Title', 'Genres']
random_movies = pd.read_csv('random_movies.csv')
movie_ids_subset = [f"m{mid}" for mid in random_movies['MovieID']]
# Create RMat from ratings.dat
ratings_copy = ratings.copy()
# Prefix movie IDs with "m" and user IDs with "u"
ratings_copy['MovieID'] = 'm' + ratings_copy['MovieID'].astype(str)
ratings_copy['UserID'] = 'u' + ratings_copy['UserID'].astype(str)
# Pivot the table to create the RMat format
RMat = ratings_copy.pivot(index='UserID', columns='MovieID', values='Rating')
# Replace NaN with "NA" for missing ratings
RMat = RMat.fillna("NA")

movieList = [
    {
        "MovieID": row.MovieID,
        "Image": f"https://liangfgithub.github.io/MovieImages/{row.MovieID}.jpg?raw=true",
        "Title": row.Title
    }
    for _, row in random_movies.iterrows()
]

app = dash.Dash(__name__)

def createMovieRows(movies, moviesPerRow = 5):
    rows = []
    for i in range(0, len(movies), moviesPerRow):
        row = html.Div(
            [
                html.Div(
                    [
                        html.Img(
                            src = movie['Image'], 
                            style = {"width": "150px", "height": "225px", "marginBottom": "10px"}
                        ),
                        html.P(movie['Title'], style = {"textAlign": "center", "fontSize": "14px", "marginTop": "5px"}),
                        dcc.RadioItems(
                            options = [{"label": str(i), "value": i} for i in range(6)],
                            id = {"type": "rating", "movie_id": movie['MovieID']},
                            className = "text-center",
                            style = {"display": "flex", "justifyContent": "center", "gap": "10px"}  
                        )

                    ],
                    style = {
                        "display": "inline-block", 
                        "margin": "20px", 
                        "textAlign": "center", 
                        "width": "180px"  
                    }
                )
                for movie in movies[i:i+moviesPerRow]
            ],
            style = {
                "display": "flex", 
                "justifyContent": "space-evenly", 
                "alignItems": "center", 
                "marginBottom": "30px"
            }
        )
        rows.append(row)
    return rows

app.layout = html.Div(
    children = [
        html.H1("Movie Recommender System", style = {'textAlign': 'center', "marginBottom": "30px"}),
        html.Div(createMovieRows(movieList)),
        html.Div(
            html.Button("Submit Ratings", id = "submit-button", n_clicks = 0, disabled = True),
            style = {"textAlign": "center", "marginTop": "20px"}
        ),
        html.Div(id = "rating-output", style = {"textAlign": "center", "marginTop": "20px"})
    ],
    style={"backgroundColor": "#f9f9f9", "padding": "20px"}
)

@app.callback(
    Output("submit-button", "disabled"),
    Input({"type": "rating", "movie_id": ALL}, "value")
)
def enable_submit_button(ratings):
    return not any(ratings)

@app.callback(
    Output("rating-output", "children"),
    Input("submit-button", "n_clicks"),
    State({"type": "rating", "movie_id": ALL}, "value"),
    State({"type": "rating", "movie_id": ALL}, "id"),
    prevent_initial_call=True,
)
def handle_submit(n_clicks, ratings, ids):
    # Step 1: Prepare new user vector for the selected movies
    newuser = pd.Series(np.nan, index=movie_ids_subset)  # Use the precomputed subset of MovieIDs
    for id, rating in zip(ids, ratings):
        if rating is not None:
            newuser[f"m{id['movie_id']}"] = rating

    # Step 2: Generate recommendations using myIBCF with the precomputed matrices
    recommendations = myIBCF(newuser, RMat[movie_ids_subset], sMatrix, top_n=10)

    # Step 3: Prepare the output
    recommendations_output = []
    for mid in recommendations:
        movie_id = int(mid.lstrip("m"))
        movie_row = random_movies[random_movies['MovieID'] == movie_id]
        if not movie_row.empty:
            title = movie_row['Title'].values[0]
            image_url = f"https://liangfgithub.github.io/MovieImages/{movie_id}.jpg"
            recommendations_output.append(
                html.Div(
                    [
                        html.Img(src=image_url, style={"width": "100px", "height": "150px"}),
                        html.P(title, style={"textAlign": "center", "fontSize": "14px", "marginTop": "5px"}),
                    ],
                    style={"marginBottom": "20px", "textAlign": "center"}
                )
            )

    return html.Div([
        html.H3("Recommended Movies:"),
        html.P(recommendations_output)
    ])

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=False)