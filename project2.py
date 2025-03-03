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
# Ensure rating stays within [1,5]
# Hybrid similarity as the average of both
def hybrid_similarity(ratings1, ratings2):
    cos_sim = cosine_similarity(ratings1, ratings2)
    pearson_sim = pearson_similarity(ratings1, ratings2)
    pearson_norm = (pearson_sim + 1) / 2.0
    return (cos_sim + pearson_norm) / 2.0

def predict_rating(active_user_known, movie, training_data, k=10, similarity_metric="cosine"):
    similarities = []
    
    # Compute mean rating of the active user
    mean_active = sum(active_user_known.values()) / len(active_user_known) if active_user_known else 3

    for user_id, user_ratings in training_data.items():
        if movie in user_ratings:
            # Choose similarity metric
            if similarity_metric == "cosine":
                sim = cosine_similarity(active_user_known, user_ratings)
            elif similarity_metric == "pearson":
                sim = pearson_similarity(active_user_known, user_ratings)
            elif similarity_metric == "hybrid":
                sim = hybrid_similarity(active_user_known, user_ratings)
            else:
                raise ValueError("Invalid similarity metric. Choose 'cosine', 'pearson', or 'hybrid'.")

            if sim > 0.1:  # Ignore very weak similarities
                mean_user = sum(user_ratings.values()) / len(user_ratings)
                adjusted_rating = user_ratings[movie] - mean_user  # Normalize rating
                similarities.append((sim, adjusted_rating))

    if not similarities:
        return 3  # Fallback to the default rating (safe integer)

    similarities.sort(reverse=True, key=lambda x: x[0])
    top_neighbors = similarities[:k]

    numerator = sum(sim * rating for sim, rating in top_neighbors)
    denominator = sum(abs(sim) for sim, _ in top_neighbors)

    if denominator == 0:
        return round(mean_active)  # Avoid NaN values

    predicted_rating = mean_active + (numerator / denominator)  # Denormalize rating
    
    # Ensure the rating is within valid range [1, 5]
    return min(5, max(1, round(predicted_rating)))


def predict_test_file(test_url, training_data, output_filename, similarity_metric="cosine"):
    print(f"Downloading test data from {test_url}...")
    test_data = load_data_from_url(test_url)
    print(f"Loaded test data for {len(test_data)} users.")

    predictions = []
    for user, ratings in test_data.items():
        # Keep only the initially known ratings (the block)
        initial_known_ratings = {movie: rating for movie, rating in ratings.items() if rating != 0}
        
        for movie, rating in ratings.items():
            if rating == 0:
                # Predict rating using only the training data and the initially known block
                pred = predict_rating(initial_known_ratings, movie, training_data, k=10, similarity_metric=similarity_metric)
                predictions.append((user, movie, pred))

    # Sorting by user (primary) and movie (secondary)
    predictions.sort(key=lambda x: (x[0], x[1]))
    with open(output_filename, 'w') as f:
        for user, movie, pred in predictions:
            f.write(f"{user} {movie} {int(pred)}\n")  # Explicitly convert pred to int

    
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
    similarity_metric = input("Enter similarity metric (cosine/pearson/hybrid): ").strip().lower()
    while similarity_metric not in ["cosine", "pearson", "hybrid"]:
        print("Invalid input. Please enter 'cosine', 'pearson', or 'hybrid'.")
        similarity_metric = input("Enter similarity metric (cosine/pearson/hybrid): ").strip().lower()

    print(f"Using {similarity_metric} similarity metric...")

    for test_name, test_url in test_files.items():
        output_filename = f"result{test_name}.txt"  # Now outputs result5.txt, result10.txt, result20.txt
        predict_test_file(test_url, training_data, output_filename, similarity_metric=similarity_metric)

if __name__ == '__main__':
    main()
