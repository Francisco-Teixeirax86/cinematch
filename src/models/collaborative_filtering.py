""""
Collaborative filtering recommendation models for CineMatch
Uses the tastes and rates of other users to recommend movies
"""
from wsgiref.util import request_uri

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from pathlib import Path
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

logger = logging.getLogger(__name__)


class UserBasedCF:
    """"
        User-Based Collaborative Filtering recommendation model
    """

    def __init__(self, user_movie_matrix):
        """"
            Initialize the user based collaborative filtering recommendation model

            Arguments:
                user_movie_matrix (pandas.DataFrame): User-movie rating matrix with
                                                        users as indices and movies as columns
        """
        self.user_movie_matrix = user_movie_matrix
        self.user_similarity_matrix = None

    def compute_user_similarity(self):
        """
                Compute similarity between users.

                Returns:
                    pandas.DataFrame: User similarity matrix.
        """
        logger.info("Computing user similarity matrix...")

        self.user_similarity_matrix = self.user_movie_matrix.T.corr(method='pearson')

        logger.info("Finished computing user similarity matrix.")
        return self.user_similarity_matrix

    def recommend_movies(self, user_id, n_recommendations=10, min_rating=3.5, similarity_threshold=0.3):
        """"
            Recommend movie for a user

            Arguments:
                user_id (int): Target User ID
                n_recommendations (int): Number of recommendations to return
                min_rating (float): Minimum recommendation rating
                similarity_threshold (float): Similarity threshold for considering users

            Returns:
                pandas.DataFrame: Recommended movie for a user
        """

        if self.user_similarity_matrix is None:
            logger.info("User similarity matrix not computed, computing user similarity matrix...")
            self.compute_user_similarity()

        logger.info(f"Generating recommendations for user {user_id}...")

        #Get movies the user has already rated
        user_rating = self.user_movie_matrix.loc[user_id]
        rated_movies = user_rating.dropna().index.toList()

        #Find similar users
        similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)
        similar_users = similar_users[similar_users >= similarity_threshold]

        #Filter out the target user
        similar_users = similar_users.drop(user_id, errors='ignore')

        if len(similar_users) == 0:
            logger.warning(f"No similar users for user {user_id}")
            return pd.DataFrame()

        #Find movies that similar users have rated but target user hasn't
        recommendations = {}

        for sim_user, similarity in similar_users.items():
            #Get ratings by similar user
            sim_user_rating = self.user_movie_matrix.loc[sim_user]


            #Consider only movies not rated by target user and with ratings >= min_rating
            for movie in sim_user_rating.index:
                rating = sim_user_rating[movie]

                if pd.notna(rating) and rating >= min_rating and movie not in rated_movies:
                    if movie not in recommendations:
                        recommendations[movie] = {'weighted_sum': 0, 'similarity_sum': 0}

                    #Weighted sum of ratings
                    recommendations[movie]['weighted_sum'] += rating * similarity
                    recommendations[movie]['similarity_sum'] += similarity

        #Calculate predicted ratings
        predicted_ratings = {}
        for movie, values in recommendations.items():
            if values['similarity_sum'] > 0:
                predicted_ratings[movie] = values['weighted_sum'] / values['similarity_sum']

        #Convert to DataFrame and sort by predicted rating
        recommendations_df = pd.DataFrame({
            'movieId': list(predicted_ratings.keys()),
            'predicted_rating': list(predicted_ratings.values())
        })

        #Sort by predicted score desc
        recommendations_df = recommendations_df.sort_values(by=['predicted_rating'], ascending=False)

        #Limit to n_recommendations
        recommendations_df = recommendations_df.head(n_recommendations)

        logger.info(f"Generated recommendations for user {user_id}")

        return recommendations_df

    def evaluate(self, test_data):
        """"
        Evaluate recommendation model.

        Arguments:
            test_data (pandas.DataFrame): Test data with columns 'userId', 'movieId' and 'rating'.

        Returns:
            float: Root Mean Squared Error of predictions.
        """

        logger.info("Evaluating recommendation model...")

        if self.user_similarity_matrix is None:
            logger.info("User similarity matrix not computed, computing user similarity matrix...")
            self.compute_user_similarity()

        squared_error = []

        for _, row in test_data.iterrows():
            user_id = row['userId']
            movie_id = row['movieId']
            actual_rating = row['rating']

            #Skip if user or movie not in training data
            if user_id not in self.user_similarity_matrix or movie_id not in self.user_similarity_matrix:
                continue

            #Get Similar users who have rated this movie
            similar_users = self.user_similarity_matrix[user_id].sort_values(ascending=False)
            similar_users = similar_users.drop(user_id, errors='ignore')

            #Filter to users who have rated this movie
            movie_rates = []

            for sim_user in similar_users.index:
                if sim_user in self.user_movie_matrix and pd.notna(self.user_movie_matrix.loc[sim_user, movie_id]):
                    movie_rates.append(sim_user)

            if not movie_rates:
                continue

            weighted_sum = 0
            similarity_sum = 0

            for rater in movie_rates:
                similarity = similar_users[rater]
                rating = self.user_movie_matrix.loc[rater, movie_id]

                weighted_sum += rating * similarity
                similarity_sum += abs(similarity)

            if similarity_sum == 0:
                continue

            predicted_rating = weighted_sum / similarity_sum

            error = predicted_rating - actual_rating
            squared_error.append(error ** 2)

        if not squared_error:
            logger.warning(f"No predictions made, cannot calculate RMSE")
            return None

        rmse = np.sqrt(np.mean(squared_error))
        logger.info(f"RMSE: {rmse}")

        return rmse
    
