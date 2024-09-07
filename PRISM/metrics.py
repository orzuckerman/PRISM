import math
from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def hit_ratio(y, pred, n=10):
    mask = np.zeros_like(y)
    mask[y > 0] = 1
    pred_masked = pred * mask
    best_index = np.argmax(y)
    pred_masked_indexes = np.argsort(pred_masked)[::-1][:n]
    if best_index in pred_masked_indexes:
        return 1
    else:
        return 0


def evaluate_hit_ratio(y_true, y_pred, k=10):
    y_pred = np.argsort(y_pred)[::-1][:k]
    return 1 if np.argmax(y_true) in y_pred else 0


def ndcg(y, pred, n=10):
    actual_recommendation_best_10indexes = np.argsort(y)[::-1][:n]
    actual_recommendation_best_10 = y[actual_recommendation_best_10indexes]
    predicted_recommendation_best_10 = pred[actual_recommendation_best_10indexes]
    predicted_recommendation_best_10 = np.around(predicted_recommendation_best_10)
    predicted_recommendation_best_10[predicted_recommendation_best_10 < 0] = 0
    dcg_numerator = np.power(2, predicted_recommendation_best_10) - 1
    denomimator = np.log2(np.arange(start=2, stop=n + 2))
    idcg_numerator = np.power(2, actual_recommendation_best_10) - 1
    dcg = np.sum(dcg_numerator / denomimator)
    idcg = np.sum(idcg_numerator / denomimator)
    if idcg != 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    return ndcg


def evaluate_ndcg(y_true, y_pred, k=10):
    y_pred = np.argsort(y_pred)[::-1][:k]
    true_index = np.argmax(y_true)
    if true_index in y_pred:
        rank = np.where(y_pred == true_index)[0][0]
        return 1 / np.log2(rank + 2)
    return 0


def compute_metrics(y, pred, metric_functions=None):
    if metric_functions is None:
        metric_functions = [hit_ratio, ndcg]
    return [fun(y, pred) for fun in metric_functions]


def calculate_cosine_similarity(users_liked_i, users_liked_j):
    intersection = len(users_liked_i.intersection(users_liked_j))
    if intersection == 0:
        return 0
    magnitude_i = math.sqrt(len(users_liked_i))
    magnitude_j = math.sqrt(len(users_liked_j))
    similarity = intersection / (magnitude_i * magnitude_j)
    return similarity


def calculate_diversity(recommendation_lists, test_items, item_embeddings):
    diversity_sum = 0
    total_users = len(recommendation_lists)

    for user_item_id, items in recommendation_lists.items():
        test_item = test_items[user_item_id]

        for item in items:
            item_and_test = [item, test_item]
            recommended_item_embeddings = item_embeddings[item_and_test]
            similarity_matrix = cosine_similarity(recommended_item_embeddings.detach().numpy())
            distance_matrix = 1 - similarity_matrix
            diversity_sum += np.mean(distance_matrix)

    average_diversity = diversity_sum / (total_users * 10)
    return average_diversity


def calculate_novelty(recommendation_lists, interactions_data):
    item_interactions = defaultdict(int)
    total_interactions = 0

    # Calculate item interactions based on interactions data
    for user_id, items in interactions_data.items():
        for item in items:
            item_interactions[item] += 1
            total_interactions += 1

    # Compute normalized item popularity
    item_popularity = {}
    for item, count in item_interactions.items():
        item_popularity[item] = count / total_interactions if total_interactions != 0 else 0

    total_recommendations = sum(len(items) for items in recommendation_lists.values())
    novelty_sum = 0

    for user_id, recommendation_list in recommendation_lists.items():
        user_novelty_sum = 0
        num_recommendations = len(recommendation_list)

        for item in recommendation_list:
            item_pop = item_popularity.get(item, 0)
            # Add a small constant to avoid log(0)
            item_novelty = -math.log2(item_pop + 1e-10)
            user_novelty_sum += item_novelty

        user_novelty = user_novelty_sum / num_recommendations if num_recommendations != 0 else 0
        novelty_sum += user_novelty

    novelty = novelty_sum / total_recommendations if total_recommendations != 0 else 0
    return novelty
