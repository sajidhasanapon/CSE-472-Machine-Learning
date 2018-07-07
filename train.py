import numpy as np
from math import sqrt

def train(ratings_train: np.ndarray, n_factors: int, lambd: float, n_iter: int=200):

    X = ratings_train.copy()
    X[X == -1] = 0 # masking -1's with zeroes

    V = np.random.rand(X.shape[1], n_factors)

    for i in range(n_iter):
        U = predict_user_factors(ratings=X, item_factors=V, lambd=lambd)
        V = predict_item_factors(ratings=X, user_factors=U, lambd=lambd)
        cost = get_cost(ratings=ratings_train, user_factors=U, item_factors=V)
        print(cost)

# def predict_user_factors(ratings: np.ndarray, item_factors: np.ndarray, lambd: float):
#
#     if len(ratings.shape) != 2:
#         print("X does not have correct shape.")
#         print("X must be a 2D array, even if there is a single user.")
#         print("If your X has a shape of (n,), try X = np.array([X]) and then pass X.")
#         exit(1)
#
#     X = ratings.copy()
#     X[X == -1] = 0
#
#     U = np.zeros((X.shape[0], item_factors.shape[1]))
#     right = X.dot(item_factors)
#     for i in range(U.shape[0]):
#         temp = item_factors[X[i] != 0]
#         left = np.linalg.inv(temp.T.dot(temp) + lambd * np.eye(item_factors.shape[1]))
#         U[i] = left.dot(right[i])
#     return U
#
# def predict_item_factors(ratings: np.ndarray, user_factors: np.ndarray, lambd: float):
#     return predict_user_factors(ratings.T, user_factors, lambd)
#
#     # this works because if X = UV then X.T = (V.T)(U.T)


# def predict_user_factors(ratings, item_factors, lambd):
#     YTY = item_factors.T.dot(item_factors)
#     lambdaI = np.eye(YTY.shape[0]) * lambd
#
#     U = np.zeros((ratings.shape[0], item_factors.shape[1]))
#     for i in range(U.shape[0]):
#         U[i, :] = np.linalg.solve((YTY + lambdaI), ratings[i, :].dot(item_factors))
#     return U
#
# def predict_item_factors(ratings: np.ndarray, user_factors: np.ndarray, lambd: float):
#     XTX = user_factors.T.dot(user_factors)
#     lambdaI = np.eye(XTX.shape[0]) * lambd
#
#     V = np.zeros((ratings.shape[1], user_factors.shape[1]))
#     for i in range(V.shape[0]):
#         V[i, :] = np.linalg.solve((XTX + lambdaI),ratings[:, i].T.dot(user_factors))
#     return V

def predict_user_factors(ratings: np.ndarray, item_factors: np.ndarray, lambd):

    if len(ratings.shape) != 2:
        print("X does not have correct shape.")
        print("X must be a 2D array, even if there is a single user.")
        print("If your X has a shape of (n,), try X = np.array([X]) and then pass X.")
        exit(1)

    X = ratings.copy()
    X[X == -1] = 0
    V = item_factors.copy()

    # U = np.zeros((X.shape[0], V.shape[1]))

    left  = np.linalg.inv(V.T.dot(V) + np.eye(V.shape[1]) * lambd)
    right = X.dot(V)

    return left.dot(right.T).T

    # for i in range(U.shape[0]):
    #     right = a[i]
    #     b = V[X[i] != 0]
    #     c = b.T.dot(b) + np.eye()
    #     first  = np.zeros((U.shape[1], U.shape[1]))
    #     second = np.zeros((U.shape[1], U.shape[1]))

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