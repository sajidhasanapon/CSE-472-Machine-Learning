import pandas as pd
from methods import *
import pickle

X_train = pd.read_excel("ratings_train.xlsx", header=None).as_matrix()
# X_validate = pd.read_excel("ratings_validate.xlsx", header=None).as_matrix()
# X_test = pd.read_excel("ratings_test.xlsx", header=None).as_matrix()

# no separate test-dataset was provided.
# Hence, the validation-dataset was split into validation-dataset and test-dataset

X_validate_and_test = pd.read_excel("ratings_validate.xlsx", header=None).as_matrix()
X_validate = X_validate_and_test[0:1999] # better : [0 : int(X_validate_and_test.shape[0]/2)]
X_test = X_validate_and_test[2000:] # better : [int(X_validate_and_test.shape[0]/2)+1 : ]



best_V = None
best_lambd = None
best_k = None
prev_cost = float("inf")

print("Training and validating...\n")
print("Scores: \n")

for k in [10, 20, 40]:
    for lambd in [0.01, 0.1, 1, 10.0]:
        V = train(ratings_train=X_train, n_factors=k, lambd=lambd, n_iter=1)
        U = predict_user_factors(X_train, V, lambd)
        cost_train = get_cost(X_train, U, V)

        U = predict_user_factors(X_validate, V, lambd)
        cost_validate = get_cost(X_validate, U, V)

        print("k = %d\t\tlamda = %f\t\tcost_train    = %f" % (k, lambd, cost_train))
        print("k = %d\t\tlamda = %f\t\tcost_validate = %f\n" %(k, lambd, cost_validate))

        if cost_validate < prev_cost:
            prev_cost = cost_validate
            best_k = k
            best_lambd = lambd
            best_V = V


# for test / recommendation without training+validating, i.e., standalone testing
pickle.dump((best_lambd, best_V), open("model", "wb"), protocol=pickle.HIGHEST_PROTOCOL)


# human readable format of the model
np.savetxt("best_V.csv", best_V, delimiter=",")
print(best_lambd, file=open("best_lambda.txt", "w"))


U = predict_user_factors(X_test, best_V, best_lambd)
cost = get_cost(X_test, U, best_V)
print("\n\n##################################################################################\n\n")
print("Cost = ", cost)
