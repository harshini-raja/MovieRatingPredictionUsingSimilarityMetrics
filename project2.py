#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:56:40 2025

@author: Harshini Raja
"""

import math
import requests
from collections import defaultdict

def load_data_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raises an error if the download failed
    data = defaultdict(dict)
    lines = response.text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue  # Skip malformed lines
        user, movie, rating = parts[:3]
        user = int(user)
        movie = int(movie)
        rating = float(rating)
        data[user][movie] = rating
    return data

def cosine_similarity(ratings1, ratings2):
    common_movies = set(ratings1.keys()) & set(ratings2.keys())
    if not common_movies:
        return 0.0

    dot_product = sum(ratings1[m] * ratings2[m] for m in common_movies)
    norm1 = math.sqrt(sum(ratings1[m] ** 2 for m in common_movies))
    norm2 = math.sqrt(sum(ratings2[m] ** 2 for m in common_movies))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def pearson_similarity(ratings1, ratings2):
    common_movies = set(ratings1.keys()) & set(ratings2.keys())
    if not common_movies:
        return 0.0

    mean1 = sum(ratings1[m] for m in common_movies) / len(common_movies)
    mean2 = sum(ratings2[m] for m in common_movies) / len(common_movies)

    num = sum((ratings1[m] - mean1) * (ratings2[m] - mean2) for m in common_movies)
    den1 = math.sqrt(sum((ratings1[m] - mean1) ** 2 for m in common_movies))
    den2 = math.sqrt(sum((ratings2[m] - mean2) ** 2 for m in common_movies))

    if den1 == 0 or den2 == 0:
        return 0.0

    return num / (den1 * den2)

# Hybrid similarity as the average of both
def hybrid_similarity(ratings1, ratings2):
    cos_sim = cosine_similarity(ratings1, ratings2)
    pearson_sim = pearson_similarity(ratings1, ratings2)
    pearson_norm = (pearson_sim + 1) / 2.0
    return (cos_sim + pearson_norm) / 2.0


def predict_rating(active_user_known, movie, training_data, k=10, similarity_metric="cosine", item_similarities=None,rho=2.5):
    """ Predict movie rating for a user using the selected similarity metric. """
    similarities = []
    
    # Compute mean rating of the active user
    mean_active = sum(active_user_known.values()) / len(active_user_known) if active_user_known else 3

    # Item-Based Cosine Similarity uses the precomputed matrix
    if similarity_metric == "item-based-cosine":
        if item_similarities is None or movie not in item_similarities:
            return round(mean_active)  # No similarity data available

        for known_movie, known_rating in active_user_known.items():
            if known_movie in item_similarities[movie]:
                sim = item_similarities[movie][known_movie]
                similarities.append((sim, known_rating))
    
    else:  # User-Based Collaborative Filtering
        for user_id, user_ratings in training_data.items():
            if movie in user_ratings:
                # Select similarity metric
                if similarity_metric == "cosine":
                    sim = cosine_similarity(active_user_known, user_ratings)
                elif similarity_metric == "pearson":
                    sim = pearson_similarity(active_user_known, user_ratings)
                elif similarity_metric == "hybrid":
                    sim = hybrid_similarity(active_user_known, user_ratings)
                elif similarity_metric == "pearson-iuf":
                    sim = pearson_iuf_similarity(active_user_known, user_ratings, training_data)
                elif similarity_metric == "pearson-case":
                    sim = pearson_case_amplification_similarity(active_user_known, user_ratings, rho=rho)
                else:
                    raise ValueError("Invalid similarity metric. Choose 'cosine', 'pearson', 'hybrid', 'pearson-iuf', 'pearson-case', or 'item-based-cosine'.")

                if sim > 0.1:  # Ignore weak similarities
                    mean_user = sum(user_ratings.values()) / len(user_ratings)
                    adjusted_rating = user_ratings[movie] - mean_user  # Normalize rating
                    similarities.append((sim, adjusted_rating))

    if not similarities:
        return 3  # Fallback default rating

    similarities.sort(reverse=True, key=lambda x: x[0])  # Sort by similarity
    top_neighbors = similarities[:k]  # Select top-k similar movies/users

    numerator = sum(sim * rating for sim, rating in top_neighbors)
    denominator = sum(abs(sim) for sim, _ in top_neighbors)

    if denominator == 0:
        return round(mean_active)  # Avoid NaN values

    predicted_rating = mean_active + (numerator / denominator)  # Denormalize rating

    return min(5, max(1, round(predicted_rating))) 
 # Ensure valid rating range
def item_based_cosine_similarity(training_data):
    """ Precompute item-item cosine similarity matrix using user ratings. """
    item_ratings = defaultdict(dict)

    # Convert user-based data to item-based data
    for user, movies in training_data.items():
        for movie, rating in movies.items():
            item_ratings[movie][user] = rating

    item_similarities = defaultdict(dict)

    # Compute cosine similarity between all pairs of items
    for item1 in item_ratings:
        for item2 in item_ratings:
            if item1 == item2:
                continue
            sim = cosine_similarity(item_ratings[item1], item_ratings[item2])
            if sim > 0:  # Store only meaningful similarities
                item_similarities[item1][item2] = sim

    return item_similarities  # Return precomputed item-item similarity matrix


def pearson_iuf_similarity(ratings1, ratings2, training_data):
    """
    Compute Pearson Correlation Coefficient with Inverse User Frequency (IUF).
    Uses the IUF formula from the image: IUF(j) = log(m / m_j).
    """
    common_movies = set(ratings1.keys()) & set(ratings2.keys())
    if not common_movies:
        return 0.0

    # Compute IUF for each movie in the common set
    total_users = len(training_data)  # m: total number of users
    movie_iuf = {}

    for movie in common_movies:
        user_count = sum(1 for user in training_data if movie in training_data[user])  # m_j: users who rated movie j
        if user_count > 0:
            movie_iuf[movie] = math.log(total_users / user_count)  # Using formula from image
        else:
            movie_iuf[movie] = 0  # Avoid log(0) error

    # Compute weighted mean ratings
    mean1 = sum(ratings1[m] * movie_iuf[m] for m in common_movies) / sum(movie_iuf.values())
    mean2 = sum(ratings2[m] * movie_iuf[m] for m in common_movies) / sum(movie_iuf.values())

    # Compute Pearson correlation with IUF weighting
    num = sum((ratings1[m] - mean1) * (ratings2[m] - mean2) * movie_iuf[m] for m in common_movies)
    den1 = math.sqrt(sum(((ratings1[m] - mean1) ** 2) * movie_iuf[m] for m in common_movies))
    den2 = math.sqrt(sum(((ratings2[m] - mean2) ** 2) * movie_iuf[m] for m in common_movies))

    if den1 == 0 or den2 == 0:
        return 0.0

    return num / (den1 * den2)


def pearson_case_amplification_similarity(ratings1, ratings2, rho=2.5):
    common_movies = set(ratings1.keys()) & set(ratings2.keys())
    if not common_movies:
        return 0.0

    # Compute mean ratings
    mean1 = sum(ratings1[m] for m in common_movies) / len(common_movies)
    mean2 = sum(ratings2[m] for m in common_movies) / len(common_movies)

    # Compute Pearson correlation
    num = sum((ratings1[m] - mean1) * (ratings2[m] - mean2) for m in common_movies)
    den1 = math.sqrt(sum((ratings1[m] - mean1) ** 2 for m in common_movies))
    den2 = math.sqrt(sum((ratings2[m] - mean2) ** 2 for m in common_movies))

    if den1 == 0 or den2 == 0:
        return 0.0

    pearson_sim = num / (den1 * den2)

    # Apply Case Amplification
    amplified_sim = pearson_sim * (abs(pearson_sim) ** (rho - 1))

    return amplified_sim

    

def predict_test_file(test_url, training_data, output_filename, similarity_metric="cosine", item_similarities=None,rho=2.5):
    print(f"Downloading test data from {test_url}...")
    test_data = load_data_from_url(test_url)
    print(f"Loaded test data for {len(test_data)} users.")

    predictions = []
    for user, ratings in test_data.items():
        initial_known_ratings = {movie: rating for movie, rating in ratings.items() if rating != 0}
        
        for movie, rating in ratings.items():
            if rating == 0:
                # Predict rating using the chosen similarity metric
                pred = predict_rating(
                    initial_known_ratings, movie, training_data, k=10,
                    similarity_metric=similarity_metric, item_similarities=item_similarities,rho=rho
                )
                predictions.append((user, movie, pred))

    # Sort predictions by user and movie
    predictions.sort(key=lambda x: (x[0], x[1]))
    with open(output_filename, 'w') as f:
        for user, movie, pred in predictions:
            f.write(f"{user} {movie} {int(pred)}\n")

    print(f"Predictions saved to {output_filename}")


def main():
    train_url = 'https://www.cse.scu.edu/~yfang/csen272/train.txt'
    test_files = {
        '5': 'https://www.cse.scu.edu/~yfang/csen272/test5.txt',
        '10': 'https://www.cse.scu.edu/~yfang/csen272/test10.txt',
        '20': 'https://www.cse.scu.edu/~yfang/csen272/test20.txt'
    }

    print("Downloading training data...")
    training_data = load_data_from_url(train_url)
    print(f"Loaded training data for {len(training_data)} users.")

   # Asking user for similarity metric choice
    similarity_metric = input("Enter similarity metric (cosine/pearson/hybrid/pearson-iuf/pearson-case/item-based-cosine): ").strip().lower()
    while similarity_metric not in ["cosine", "pearson", "hybrid", "pearson-iuf","pearson-case", "item-based-cosine"]:
        print("Invalid input. Please enter 'cosine', 'pearson', 'hybrid', 'pearson-iuf','pearson-case' or 'item-based-cosine'.")
        similarity_metric = input("Enter similarity metric (cosine/pearson/hybrid/pearson-iuf/pearson-case/item-based-cosine): ").strip().lower()
        
    # Ask for rho value if Pearson Case Amplification is selected
    rho = 2.5  # Default value
    if similarity_metric == "pearson-case":
        try:
            rho = float(input("Enter case amplification power (ρ ≥ 1, default = 2.5): ").strip())
            if rho < 1:
                print("ρ must be ≥ 1. Using default value 2.5.")
                rho = 2.5
        except ValueError:
            print("Invalid input. Using default value 2.5.")
            rho = 2.5

    print(f"Using {similarity_metric} similarity metric...")
    item_similarities = item_based_cosine_similarity(training_data) if similarity_metric == "item-based-cosine" else None

    for test_name, test_url in test_files.items():
        output_filename = f"result{test_name}.txt"  # Now outputs result5.txt, result10.txt, result20.txt
        predict_test_file(test_url, training_data, output_filename, similarity_metric=similarity_metric)

if __name__ == '__main__':
    main()
