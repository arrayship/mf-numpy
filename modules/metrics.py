import numpy as np

def rmse(ratings, ratings_pred):
    sum = 0
    for r, r_pred in zip(ratings, ratings_pred):
        sum += (r - r_pred) ** 2
    return np.sqrt(sum / len(ratings))