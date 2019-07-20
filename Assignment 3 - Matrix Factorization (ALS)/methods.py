import numpy as np
from math import sqrt



def train(ratings_train: np.ndarray, n_latent_factors: int, lambd: float, n_iter: int=10):

    X = ratings_train.copy()
    X[X == -1] = 0 # masking -1's with zeroes

    V = np.random.rand(X.shape[1], n_latent_factors)

    for i in range(n_iter):
        U = predict_user_factors(ratings=X, item_factors=V, lambd=lambd)
        V = predict_item_factors(ratings=X, user_factors=U, lambd=lambd)
        # cost = get_cost(ratings=ratings_train, user_factors=U, item_factors=V)
        # print(cost)

    return U, V


def predict_user_factors(ratings: np.ndarray, item_factors: np.ndarray, lambd: float):

    if len(ratings.shape) != 2:
        print("X does not have correct shape.")
        print("X must be a 2D array, even if there is a single user.")
        print("If your X has a shape of (n,), try X = np.array([X]) and then pass X.")
        exit(1)

    X = ratings
    V = item_factors
    k = V.shape[1]

    right = np.matmul(X, V) # named "right" according to instruction document

    left = np.matmul(V.T, V) + lambd * np.eye(k)
    left_inv = np.linalg.inv(left)

    U = right.dot(left_inv) # note the reverse order of multiplication: right x left
    return U

    """
    I can get the whole U matrix just by transposing and altering the order of matrix multiplication.
    No need to iterate over rows.

    It took me hours to realize this simple vectorization trick.
    Perhaps because BUET has murdered my love for Mathematics that I have become so naive.

    I don't know if I'll ever get back my intuition and common sense.
    I hope I will... someday.
    """


def predict_item_factors(ratings: np.ndarray, user_factors: np.ndarray, lambd: float):
    return predict_user_factors(ratings.T, user_factors, lambd)


def get_cost(ratings: np.ndarray, user_factors: np.ndarray, item_factors: np.ndarray):

    original_ratings = ratings
    prediction      = np.matmul(user_factors, item_factors.T)

    prediction[prediction < 0.] = 0.
    prediction[prediction > 5.] = 5.

    prediction[ratings == -1]         = -1 # so that difference = 0 

    diff            = original_ratings - prediction
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