import gzip
from collections import defaultdict
import numpy as np

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)

### Rating baseline: compute averages for each user, or return the global average if we've never seen the user before    

def train(users, items, k, lambda_u, lambda_v):
    U = dict()
    V = dict()
    for user in users:
        U[user] = np.random.rand(k, 1)
    for item in items:
        V[item] = np.random.rand(k, 1)

    prev_cost = float('inf')
    iter_cnt = 1
    while(True):
        for user in users:
            first = np.zeros((k, k))
            second = np.zeros((k, 1))

            r_n = users[user]
            for pair in r_n:
                m = pair[0]
                v_m = V[m]
                X_n_m = pair[1]

                first += np.matmul(v_m, v_m.T)
                second += X_n_m * v_m
            first += lambda_u * np.eye(k)
            first = np.linalg.inv(first)

            U[user] = np.matmul(first, second)

        for item in items:
            first = np.zeros((k, k))
            second = np.zeros((k, 1))
            c_m = items[item]
            for pair in c_m:
                n = pair[0]
                if n not in users:
                    continue
                u_n = U[n]
                X_n_m = pair[1]

                first += np.matmul(u_n, u_n.T)
                second += X_n_m * u_n
            first += lambda_v * np.eye(k)
            first = np.linalg.inv(first)
            V[item] = np.matmul(first, second)

        cost = 0
        for user in users:
            u = users[user]
            for pair in u:
                item = pair[0]
                x_real = pair[1]
                x_prediction = np.matmul(U[user].T, V[item])
                cost += (x_real - x_prediction)**2
        # iter_cnt +=1
        # print("Iteration : ", iter_cnt, "\tcost :", cost)

        if cost > prev_cost or (prev_cost - cost)/cost*100 < 0.1:
            print("Converged")
            file = open("stat.txt", "a")
            file.write("%2d\t%f\t%f\t%f\n" %(k, lambda_u, lambda_v, cost))
            file.close()
            return 
        prev_cost = cost

def main():
    users = defaultdict(list)
    items = defaultdict(list)
    total_size = 100

    k = [10, 20, 30, 40, 50]
    lambda_u = [0.01, 0.1, 1, 10]
    lambda_v = [0.01, 0.1, 1, 10]

    i = 0
    for l in readGz("train.json.gz"):
        user, item, rating = l['reviewerID'], l['itemID'], l['rating']
        users[user].append((item, rating))
        items[item].append((user, rating))

        i += 1
        if i >= total_size:
            break
        # print(i)

    # x = (enumerate(users))
    # for i in x:
    #     print(i)

    # l = [z for z in users]
    # print(l) 
    # exit(0)
 
    list_users              = [u for u in users]
    len_list_users          = len(list_users)
    list_users_train        = list_users[0:int(0.6*len_list_users)]
    list_users_validation   = list_users[int(0.6*len_list_users):int(0.8*len_list_users)]
    list_users_test         = list_users[int(0.8*len_list_users):]

    users_train         = {key:users[key] for key in list_users_train}
    users_validation    = {key:users[key] for key in list_users_validation}
    users_test          = {key:users[key] for key in list_users_test}

    # items_train = defaultdict(list)
    # for user in list_users_train:
    #     for pair in users_train[user]:
    #         item = pair[0]
    #         rating = pair[1]
    #         items_train[item].append((user, rating))

    # items_validation = defaultdict(list)
    # for user in list_users_validation:
    #     for pair in users_validation[user]:
    #         item = pair[0]
    #         rating = pair[1]
    #         items_validation[item].append((user, rating))

    # items_test = defaultdict(list)
    # for user in list_users_test:
    #     for pair in users_test[user]:
    #         item = pair[0]
    #         rating = pair[1]
    #         items_test[item].append((user, rating))


    for p in k:
        for q in lambda_u:
            for r in lambda_v:
                train(users_train, items, p, q, r)
        
if __name__ == "__main__":
    main()
