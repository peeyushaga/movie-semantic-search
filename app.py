from flask import Flask, render_template, request
from recommendations import movies_by_search, movies_by_movie

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def search_movies():
  if request.method == "POST":
    search_term = request.form["search_term"]
    movies = movies_by_search(search_term)  # This function should return a list of objects with 'title' and 'poster_path' keys
    return render_template("movies.html", movies=movies)
  else:
    return render_template("search.html")

@app.route("/movies")
def show_movies():
  # This route is not used in this example but can be used to display movies from another source
  pass

# @app.route("/movie-details/{movie_name}")
# def movie_details():
    
#   return movie_name

if __name__ == "__main__":
  app.run(debug=True)
