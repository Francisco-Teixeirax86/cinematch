import logging
from pathlib import Path
import os
import pandas as pd

#Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "ml-latest-small"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def load_raw_data() :
    logger.info("Loading raw data...")

    #Paths
    movies_path = RAW_DATA_DIR / "movies.csv"
    ratings_path = RAW_DATA_DIR / "ratings.csv"
    tags_path = RAW_DATA_DIR / "tags.csv"
    links_path = RAW_DATA_DIR / "links.csv"

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    tags = pd.read_csv(tags_path)
    links = pd.read_csv(links_path)

    logger.info(f"Loaded {len(movies)} movies, {len(ratings)} ratings, {len(tags)} tags")

    return movies, ratings, tags, links

def process_movies(movies):
    logger.info("Processing movies...")

    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)$')
    movies['year'] = pd.to_numeric(movies['year'], errors='coerce')

    # Clean title by removing year which is not needed
    movies['title_clean'] = movies['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)

    genre_dummies = movies['genres'].str.get_dummies(sep='|')
    movies_with_genres = pd.concat([movies, genre_dummies], axis=1)

    return movies_with_genres

def process_ratings(ratings):
    logger.info("Processing ratings...")

    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

    return ratings

def create_user_movie_matrix(ratings):
    logger.info("Creating user movie matrix...")

    user_movie_matrix = ratings.pivot(
        index='userId',
        columns='movieId',
        values='rating')
    logger.info("User movie matrix created")

    return user_movie_matrix

def main():
    """"Main preprocesing of the data"""
    logger.info("Preprocessing MovieLens Dataset...")

    if not os.path.exists(PROCESSED_DATA_DIR):
        os.makedirs(PROCESSED_DATA_DIR)

    movies, ratings, tags, links = load_raw_data()

    processed_movies = process_movies(movies)
    processed_ratings = process_ratings(ratings)
    user_movie_matrix = create_user_movie_matrix(processed_ratings)

    processed_movies.to_csv(PROCESSED_DATA_DIR / "processed_movies.csv", index=False)
    processed_ratings.to_csv(PROCESSED_DATA_DIR / "processed_ratings.csv", index=False)
    user_movie_matrix.to_csv(PROCESSED_DATA_DIR / "user_movie_matrix.csv")

    return processed_movies, processed_ratings, user_movie_matrix

if __name__ == "__main__":
    main()