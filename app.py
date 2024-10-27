from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
import uvicorn

app = FastAPI()

# Load the movies dataset
movies = pd.read_csv('movies.csv')  # Ensure this path is correct
movies_2 = movies.copy()
movies_2['genres'] = movies_2['genres'].fillna('').str.split('|')

# Create genre features
mlb = MultiLabelBinarizer()
genre_matrix = mlb.fit_transform(movies_2['genres'])

# Create the genre dataframe
genre_df = pd.DataFrame(genre_matrix, columns=mlb.classes_, index=movies_2.index)

# Combine movie information with genre encoding
movies_with_genres = pd.concat([movies, genre_df], axis=1)

# Prepare the genre matrix without unnecessary columns
movies_with_genres_short = movies_with_genres.drop(columns=['genres', 'title', 'movieId'])

# Load the trained KNN model
with open("classifier.pkl", "rb") as model_file:
    knn_model_genres = pickle.load(model_file)

class MovieRequest(BaseModel):
    movie_name: str

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Movie Recommendation System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f2f2f2;
                padding: 20px;
            }
            h1 {
                text-align: center;
            }
            label {
                font-weight: bold;
            }
            input[type="text"] {
                width: 300px;
                padding: 10px;
                margin-right: 10px;
            }
            button {
                padding: 10px 15px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
            #recommendations {
                margin-top: 20px;
                background: white;
                padding: 15px;
                border-radius: 5px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .movie {
                margin-bottom: 10px;
            }
            .movie-title {
                font-weight: bold;
            }
            .movie-genres {
                font-style: italic;
                color: #555;
            }
        </style>
    </head>
    <body>
        <h1>Movie Recommendation System</h1>
        <label for="movieInput">Enter a Movie Title:</label>
        <input type="text" id="movieInput" placeholder="e.g. Toy Story (1995)">
        <button id="recommendButton">Get Recommendations</button>

        <div id="recommendations"></div>

        <script>
            document.getElementById('recommendButton').onclick = async function() {
                const movieName = document.getElementById('movieInput').value;
                const response = await fetch('/getRecommendations', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ movie_name: movieName }),
                });

                if (response.ok) {
                    const recommendations = await response.json();
                    displayRecommendations(recommendations);
                } else {
                    const error = await response.json();
                    alert(error.detail);
                }
            };

            function displayRecommendations(recommendations) {
                const recommendationsDiv = document.getElementById('recommendations');
                recommendationsDiv.innerHTML = '<h2>Recommended Movies:</h2>';

                recommendations.forEach(movie => {
                    // Split genres string into an array
                    const genresArray = movie.genres.split('|');
                    recommendationsDiv.innerHTML += `
                        <div class="movie">
                            <span class="movie-title">${movie.title}</span> - 
                            <span class="movie-genres">${genresArray.join(', ')}</span>
                        </div>`;
                });
            }
        </script>
    </body>
    </html>
    """


@app.post("/getRecommendations")
def recommend_movies(movie_request: MovieRequest):
    movie_title = movie_request.movie_name
    try:
        # Find the index of the movie by title
        movie_index = movies_with_genres[movies_with_genres['title'] == movie_title].index[0]
        
        # Select the genre vector as a DataFrame to keep feature names consistent
        genre_vector_df = movies_with_genres_short.iloc[[movie_index]]

        # Use the genre vector to find nearest neighbors
        distances, indexes = knn_model_genres.kneighbors(genre_vector_df, n_neighbors=5, return_distance=True)
        
        # Retrieve recommended movies by their indexes
        recommended_movies = movies_with_genres.iloc[indexes.flatten()]
        return recommended_movies[['title', 'genres']].to_dict(orient='records')
    
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Movie '{movie_title}' not found in the dataset.")
    
#Run the app with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)