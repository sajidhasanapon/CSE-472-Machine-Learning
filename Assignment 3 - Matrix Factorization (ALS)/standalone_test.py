import pandas as pd
from methods import *
import pickle


print("Testing...")

# X_test = pd.read_excel("ratings_test.xlsx", header=None).values

# no separate test-dataset was provided.
# Hence, the validation-dataset was split into validation-dataset and test-dataset

X_validate_and_test = pd.read_excel("ratings_validate.xlsx", header=None).values
X_validate = X_validate_and_test[0:1999]
X_test = X_validate_and_test[2000:]



best_lambd, best_V = pickle.load(open("model.pickle", "rb"))
U = predict_user_factors(X_test, best_V, best_lambd)
cost = get_cost(X_test, U, best_V)
print("\n\n##################################################################################\n\n")
print("Cost on test data = ", cost)