""""
Evaluation metrics for the recommendation algorithms.
"""

import numpy as np
from sklearn.metrics import mean_squared_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


def rmse(pred, targets):
    """"
        Calculate the root mean square error (RMSE).

        Arguments:
            pred {array-like} -- Predicted values.
            targets {array-like} -- Ground truth values.

        Returns:
            float -- Root mean square error.
    """

    return np.sqrt(mean_squared_error(targets, pred))

def mae(pred, targets):
    """"
        Calculate the mean absolute error (MAE).

        Arguments:
            pred {array-like} -- Predicted values.
            targets {array-like} -- Ground truth values

        Returns:
            float -- Mean absolute error.
    """

    return np.mean(np.abs(np.array(targets) - np.array(pred)))

def precision_at_k(recommendations, relevant_itemns, k):
    """"
    Calculate the precision@k.

    Arguments:
        recommendations {list} -- List of recommendations items IDs.
        relevant_itemns {list} -- List of relevant (ground truth) items IDs.
        k {int} -- Number of top items to consider.

    Returns:
        float -- Precision at k value.
    """
    if len(recommendations) == 0:
        return 0.0

    #Consider only top-k items
    recommendations = recommendations[:k]

    #Count number of relevant items in recommendations
    num_relevant = len(set(recommendations) & set(relevant_itemns))

    #Caculate precision
    precision = num_relevant / min(k, len(recommendations))

    return precision

def recall_at_k(recommendations, relevant_itemns, k):
    """"
        Calculate the recall@k

        Arguments:
            recommendations {list} -- List of recommendations items IDs.
            relevant_itemns {list} -- List of relevant (ground truth) items IDs.
            k {int} -- Number of top items to consider.

        Returns:
            float -- Recall@k value.
    """

    if len(recommendations) == 0 or len(relevant_itemns) == 0:
        return 0.0

    #Consider only top-k items
    recommendations = recommendations[:k]

    #Count number of relevent items in recommendations
    num_relevant = len(set(recommendations) & set(relevant_itemns))

    recall = num_relevant / len(relevant_itemns)

    return recall


def average_precision(recommendations, relevant_itemns):
    """"
        Calculate the average precision@k.

        Arguments:
            recommendations {list} -- List of recommendations items IDs.
            relevant_itemns {list} -- List of relevant (ground truth) items IDs.

        Returns:
              float -- Average precision value.
    """

    if not relevant_itemns or not recommendations:
        return 0.0

    relevant_set = set(relevant_itemns)
    precision_sumn = 0.0
    num_relevant = 0.0

    for i, item in enumerate(recommendations):
        if item in relevant_set:
            num_relevant += 1
            precision_sumn +=  num_relevant / (i + 1)

    if not num_relevant:
        return 0.0

    return precision_sumn / min(len(relevant_itemns), len(recommendations))

def mean_average_precision(recommendations_dict, relevance_dict):
    """"
        Calculate Mean Average Precision.

        Arguments:
            recommendations_dict {dict} -- Dictionary mapping user IDs to lists of recommended items IDs.
            relevance_dict {dict} -- Dictionary mapping user IDs to lists of relevant item IDs.

        Returns:
            float -- MAP value
    """

    if not recommendations_dict:
        return 0.0

    ap_sum = 0.0
    user_count = 0

    for user_id, recommendations_dict in recommendations_dict.items():
        if user_id in relevance_dict:
            relevant_items = relevance_dict[user_id]
            ap = average_precision(recommendations_dict, relevant_items)
            ap_sum += ap
            user_count += 1

    if user_count == 0:
        return 0.0

    return ap_sum / user_count


def diversity(recommendations, item_features):
    """"
        Caculate diversity of recommendations based on item features.

        Arguments:
            recommendations {list} -- List of recommendations items IDs.
            item_features {pandas.DataFrame} -- DataFrame containing item features.

        Returns:
            float -- Diversity value (higher means more diverse).
    """

    if len(recommendations) <= 1:
        return 0.0

    #Filter features to only include recommened items
    rec_features = item_features.loc[item_features.index.isin(recommendations)]

    if rec_features.empty:
        return 0.0

    #Calcylate pairwise distances between items
    n_items = len(rec_features)
    diversity_sum = 0.0
    pair_count = 0

    for i in range(n_items):
        for j in range(i + 1, n_items):
            #Calculate Jaccard distance for categorical features
            item_i = rec_features.loc[i]
            item_j = rec_features.loc[j]

            #Simple distance: proportion of features tha differ
            distance = np.mean(item_i != item_j)

            diversity_sum += distance
            pair_count += 1

    if pair_count == 0:
        return 0.0

    return diversity_sum / pair_count

def coverage(recommendations_dict, all_items):
    """"
        Calculate catalog coverage of recommendations.

        Arguments:
            recommendations_dict {dict} -- Dictionary mapping user IDs to lists of recommended items IDs.
            all_items {pandas.DataFrame} -- DataFrame containing item features.

        Returns:
            float -- Coverage value.
    """

    if not recommendations_dict or not all_items:
        return 0.0

    #Flatten all recommendations into a single set of unique items
    recommendations_dict = set()
    for items in recommendations_dict.values():
        recommendations_dict.update(items)


    #Calculate coverage
    coverage_ratio = len(recommendations_dict) / len(all_items)

    return coverage_ratio * 100.0

def evaluate_model(model, test_data, all_items=None, item_features=None, k=10):
    """"
    Comprehensive(I think) evaluation of recommendation model.

    Arguments:
        model {object} -- Recommendation model with predict method.
        test_data {pandas.DataFrame} -- Test data with user-item-rating triplets.
        all_items {list, optional} -- List of all available items for coverage calculation.
        item_features {pandas.DataFrame, optional} -- Item features for diversity calculation.
        k {int} -- Cutoff for top-k metrics.

    Returns:
        dict -- Dictionary with evaluation metrics.

    """
    logger.info("Starting evaluation of recommendation model.")

    predictions = []
    targets = []
    user_recommendations = {}
    user_relevance = {}

    #Group test data by user for recommendation-based metrics
    for user_id, user_data in test_data.groupby("user_id"):
        #Get relevant items (items the user liked in test set)
        relevent_items = user_data[user_data['rating'] >= 4.0]['movieId'].to_List()

        #Get model recommendations for this user
        try:
            recommendations = model.recommned_movies(user_id, n_recommendations=k*2)
            recommended_items = recommendations['movieId'].to_list()

            #Store for later metrics calculation
            user_recommendations[user_id] = recommended_items
            user_relevance[user_id] = relevent_items

            #For each test item, get prediction and actual rating
            for _, row in user_data.iterrows():
                movie_id = row['movieId']
                actual_rating = row['rating']

                #Predict rating
                try:
                    predicted_rating = model.predict_rating(user_id, movie_id)

                    predictions.append(predicted_rating)
                    targets.append(actual_rating)
                except Exception as e:
                    logger.warning(f"Failed to predict rating for user {user_id}, movie {movie_id}, Error: {e}")

        except Exception as e:
            logger.warning(f"Failed to generate recommendations for user {user_id}, Error: {e}")

    #Calculate rating predictions metrics
    metrics = {}
    if predictions and targets:
        metrics['rmse'] = rmse(predictions, targets)
        metrics['mae'] = mae(predictions, targets)

    #Calculate recommendation metrics
    if user_recommendations and user_relevance:
        precision_values = []
        recall_values = []

        for user_id, recommended_items in user_recommendations.items():
            if user_id in user_relevance:
                relevant_items = user_relevance[user_id]

                if relevant_items:
                    precision = precision_at_k(recommended_items, relevant_items, k=10)
                    recall = recall_at_k(recommended_items, relevant_items, k=10)

                    precision_values.append(precision)
                    recall_values.append(recall)

        if precision_values:
            metrics['precision_at_k'] = np.mean(precision_values)
        if recall_values:
            metrics['recall_at_k'] = np.mean(recall_values)

        #Calculate MAP
        metrics['map'] = mean_average_precision(user_recommendations, user_relevance)

    if all_items is not None and user_recommendations:
        metrics['coverage'] = coverage(user_recommendations, all_items)

    if item_features is not None and user_recommendations:
        diversity_values = []

        for recommended_items in user_recommendations.values():
            if recommended_items:
                div = diversity(recommended_items, item_features)
                diversity_values.append(div)

        if diversity_values:
            metrics['diversity'] = np.mean(diversity_values)

    logger.info("Finished evaluating recommendation model.")
    return metrics