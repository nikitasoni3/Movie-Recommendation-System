from flask import Flask, request, jsonify
from flask_cors import CORS
import recommendation

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route("/recommend", methods=["POST"])
def recommend_movies_api():
    try:
        data = request.json
        movie_title = data.get("movie_title")
        user_id = data.get("user_id")
        genre = data.get("genre")
        keywords = data.get("keywords")

        if not movie_title or not user_id:
            return jsonify({"error": "Movie title and user ID are required"}), 400

        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({"error": "Invalid user ID. Must be an integer."}), 400

        recommendations = {
            "Best Recommendations": recommendation.hybrid_recommendation(movie_title, user_id),
            "Best Recommendations on the basis of genres": recommendation.recommend_by_genre(genre) if genre else [],
            "Best Recommendations on the basis of keywords": recommendation.recommend_by_keywords(keywords) if keywords else [],
        }

        return jsonify(recommendations)
    except Exception as e:
        return jsonify({"error": f"Something went wrong: {str(e)}"}), 500

@app.route("/")
def home():
    return "Welcome to the Movie Recommendation API!"

@app.route("/movies", methods=["GET"])
def get_movies():
    try:
        return jsonify(recommendation.get_movie_list())
    except Exception as e:
        return jsonify({"error": f"Failed to fetch movies: {str(e)}"}), 500

@app.route("/users", methods=["GET"])
def get_users():
    try:
        return jsonify(recommendation.get_user_list())
    except Exception as e:
        return jsonify({"error": f"Failed to fetch users: {str(e)}"}), 500

if __name__ == "__main__":
    from gunicorn.app.wsgiapp import run
    import os

    workers = 1  # Set workers to 1 to reduce memory usage
    threads = 2  # Limit threads for efficiency

    os.environ["GUNICORN_CMD_ARGS"] = f"--workers={workers} --threads={threads} --timeout=120"
    run()
