import numpy as np
from math import sqrt



def train(ratings_train: np.ndarray, n_factors: int, lambd: float, n_iter: int=10):

    np.random.seed(100)

    X = ratings_train.copy()
    X[X == -1] = 0 # masking -1's with zeroes

    V = np.random.rand(X.shape[1], n_factors)

    for i in range(n_iter):
        U = predict_user_factors(ratings=X, item_factors=V, lambd=lambd)
        V = predict_item_factors(ratings=X, user_factors=U, lambd=lambd)
        # cost = get_cost(ratings=ratings_train, user_factors=U, item_factors=V)
        # print(cost)

    return V


def predict_user_factors(ratings: np.ndarray, item_factors: np.ndarray, lambd):

    if len(ratings.shape) != 2:
        print("X does not have correct shape.")
        print("X must be a 2D array, even if there is a single user.")
        print("If your X has a shape of (n,), try X = np.array([X]) and then pass X.")
        exit(1)

    X = ratings.copy()
    X[X == -1] = 0
    V = item_factors.copy()
    m, k = X.shape[0], V.shape[1]
    U = np.zeros((m, k))
    right = X.dot(V)

    for i in range(m):
        b = V[X[i] != 0]
        left = b.T.dot(b) + lambd * np.eye(k)
        U[i] = np.linalg.solve(left, right[i])
    return U


def predict_item_factors(ratings: np.ndarray, user_factors: np.ndarray, lambd: float):
    return predict_user_factors(ratings.T, user_factors, lambd)


def get_cost(ratings: np.ndarray, user_factors: np.ndarray, item_factors: np.ndarray):

    original_rating = ratings.copy()
    prediction      = np.matmul(user_factors, item_factors.T)

    prediction[ratings == -1]         = 0
    original_rating[ratings == -1]    = 0

    prediction[prediction < 0.] = 0.
    prediction[prediction > 5.] = 5.

    diff            = original_rating - prediction
    diff_squared    = diff * diff
    total_observed  = len(ratings[ratings != -1])
    cost            = sqrt(np.sum(diff_squared) / total_observed)

    return cost











# # fastest
# def predict_user_factors(ratings: np.ndarray, item_factors: np.ndarray, lambd):
#
#     if len(ratings.shape) != 2:
#         print("X does not have correct shape.")
#         print("X must be a 2D array, even if there is a single user.")
#         print("If your X has a shape of (n,), try X = np.array([X]) and then pass X.")
#         exit(1)
#
#     X = ratings.copy()
#     X[X == -1] = 0
#     V = item_factors.copy()
#     left  = np.linalg.inv(V.T.dot(V) +  lambd * np.eye(V.shape[1]))
#     right = (X.dot(V)).T
#     return left.dot(right).T
#
# def predict_item_factors(ratings: np.ndarray, user_factors: np.ndarray, lambd: float):
#     return predict_user_factors(ratings.T, user_factors, lambd)