""""
    Script for testing and comparing recommendation models
"""

import os
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
import logging
import matplotlib.pyplot as plt

#Add parent directory to path to import local modules
sys.path.append(str(Path(__file__).parents[2]))
from src.models.collaborative_filtering import UserBasedCF, ItemBasedCF


logging.basicConfig(level=logging.INFO, format='%(asctime)s : %(levelname)s : %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(__file__).parents[2] / "results"

def load_data():
    logger.info("Loading data")

    #Load processed data
    ratings_path = PROCESSED_DATA_DIR / "processed_ratings.csv"
    ratings = pd.read_csv(ratings_path)

    #Load movies data
    movies_path = PROCESSED_DATA_DIR / "processed_movies.csv"
    movies = pd.read_csv(movies_path)

    #Load user-movie matrix
    user_movie_matrix_path = PROCESSED_DATA_DIR / "user_movie_matrix.csv"
    user_movie_matrix = pd.read_csv(user_movie_matrix_path, index_col=0)

    logger.info("Loading user and movie ratings")

    return ratings, movies, user_movie_matrix

def prepare_training_test_data(ratings, test_size=0.2, random_state = 42):
    """"Split data into training and test sets"""
    logger.info("Preparing training data")

    train_data, test_data = train_test_split(ratings, test_size=test_size, random_state=random_state)

    logger.info("Prepared test data")

    return train_data, test_data

def create_user_movie_matrix(ratings_df):
    """Create user-movie matrix from ratings dataframe"""
    logger.info("Creating user movie matrix")

    user_movie_matrix = ratings_df.pivot(index="userId", columns="movieId", values="rating")

    logger.info("Created user movie matrix")

    return user_movie_matrix

def test_user_based_cf(train_data, test_data):
    logger.info("Testing user based CF")
    
    try:
        user_movie_matrix = create_user_movie_matrix(train_data)

        model = UserBasedCF(user_movie_matrix)
        model.compute_user_similarity()

        rmse = model.evaluate(test_data)

        if rmse is None:
            logger.warning("Could not calculate RMSE for User-Based CF")
        else:
            logger.info(f"User based CF RMSE: {rmse}")

        return model, rmse

    except Exception as e:
        logger.error(f"Error in User-Based CF evaluation: {str(e)}")
        # Return the model if we have it, otherwise None, and infinity for RMSE
        return UserBasedCF(user_movie_matrix) if 'user_movie_matrix' in locals() else None, float('inf')

def test_item_based_cf(train_data, test_data):
    logger.info("Testing item based CF")

    try:
        user_movie_matrix = create_user_movie_matrix(train_data)

        model = ItemBasedCF(user_movie_matrix)
        model.compute_movie_similarity()

        rmse = model.evaluate(test_data)

        if rmse is None:
            logger.warning("Could not calculate RMSE for Item-Based CF")
        else:
            logger.info(f"Item based CF RMSE: {rmse}")

        return model, rmse

    except Exception as e:
        logger.error(f"Error in Item-Based CF evaluation: {str(e)}")
        # Return the model if we have it, otherwise None, and infinity for RMSE
        return ItemBasedCF(user_movie_matrix) if 'user_movie_matrix' in locals() else None, float('inf')

def compare_models(ratings):
    """"Compare different recommendation models"""

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    #Split data into training and test sets
    train_data, test_data = prepare_training_test_data(ratings)

    results = {}

    #User based CF pearson
    user_cf, rmse_user = test_user_based_cf(train_data, test_data)
    results["user_cf"] = rmse_user if rmse_user is not None else float('inf')

    #Item based CF pearson
    item_cf, rmse_item = test_item_based_cf(train_data, test_data)
    results["item_cf"] = rmse_item if rmse_item is not None else float('inf')

    plot_results = {}
    for model, rmse in results.items():
        if rmse == float('inf'):
            plot_results[model] = 0
        else:
            plot_results[model] = rmse

    if any(rmse != float('inf') for rmse in results.values()):
        # Plot results
        plt.figure(figsize=(10, 6))
        bars = plt.bar(plot_results.keys(), plot_results.values())

        for i, model in enumerate(plot_results.keys()):
            if results[model] == float('inf'):
                plt.text(i, 0.1, "No valid RMSE",
                         ha='center', va='bottom', rotation=90, color='red')

        plt.title("Comparison of recommendation models, lower is better")
        plt.ylabel("RMSE")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "comparison.png")
        plt.close()

    display_results = {}
    for model, rmse in results.items():
        if rmse == float('inf'):
            display_results[model] = "No valid RMSE"
        else:
            display_results[model] = rmse

    results_df = pd.DataFrame({
        'Model': list(results.keys()),
        'RMSE': list(results.values())
    })

    results_df.to_csv(RESULTS_DIR / "comparison.csv", index=False)

    logger.info(f"Results saved in {RESULTS_DIR}")

    valid_results = {k: v for k, v in results.items() if v != float('inf') and v is not None}

    if not valid_results:
        logger.warning("No valid models found. Using first available model.")
        # Use the first model that exists
        if "user_cf" in results and user_cf is not None:
            logger.info("Using user-based CF as fallback")
            return user_cf
        elif "item_cf" in results and item_cf is not None:
            logger.info("Using item-based CF as fallback")
            return item_cf
        else:
            logger.error("No working models found. Cannot proceed.")
            return None

    best_model_name = min(results, key=results.get)

    logger.info(f"Best model name: {best_model_name} with RMSE: {results[best_model_name]}")

    if best_model_name == "user_cf":
        return user_cf
    elif best_model_name == "item_cf":
        return item_cf

def generate_sample_recommendation(model, movies_df, n_users=5, n_recommendations=10):
    logger.info(f"Generating recommendations for {n_users} users")


    user_ids = np.random.choice(model.user_movie_matrix.index, size=n_users, replace=False)

    recommendations_by_user = {}

    for user_id in user_ids:
        recommendations = model.recommend_movies(user_id, n_recommendations=n_recommendations)

        if not recommendations.empty:
            user_recommendations = recommendations.merge(
                movies_df[['movieId', 'title', 'genres', 'year']],
                on='movieId',
            )

            recommendations_by_user[user_id] = user_recommendations

    for user_id, user_recommendations in recommendations_by_user.items():
        filename = f"{user_id}_recommendations.csv"
        user_recommendations.to_csv(RESULTS_DIR / filename, index=False)


    logger.info(f"Recommendations saved in {RESULTS_DIR}")

    return recommendations_by_user

def main():
    """"Main function"""
    logger.info("Starting model testing")

    ratings, movies, _ = load_data()

    best_model = compare_models(ratings)

    recommendations = generate_sample_recommendation(best_model, movies)

    for user_id, user_recommendations in recommendations.items():
        print(f"\nRecommendations for user {user_id}:")
        print(user_recommendations[['title', 'year', 'genres', 'predicted_rating']].head(5))
        break #Just show the first user's recommendations

    logger.info("Model testing complete")


if __name__ == "__main__":
    main()
