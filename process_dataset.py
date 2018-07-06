import gzip
from collections import defaultdict
import pandas as pd
import operator
import pickle

##################
total_size = 10000
##################

def readGz(f):
    for l in gzip.open(f):
        yield eval(l)


count = defaultdict(list)

for l in readGz("train.json.gz"):
    user = l['reviewerID']
    if user in count:
        count[user] += 1
    else:
        count[user] = 1

top_users = sorted(count.items(), key=operator.itemgetter(1), reverse=True)[0:10000]
list_users = [pair[0] for pair in top_users]

list_users_train        = list_users[0:int(0.6*total_size)]
list_users_validation   = list_users[int(0.6*total_size):int(0.8*total_size)]
list_users_test         = list_users[int(0.8*total_size):]

users_train         = defaultdict(list)
users_validation    = defaultdict(list)
users_test          = defaultdict(list)
items               = defaultdict(list)

for l in readGz("train.json.gz"):
    user, item, rating = l['reviewerID'], l['itemID'], l['rating']
    if user in list_users_train:
        users_train[user].append((item, rating))
        items[item].append((user, rating))
    elif user in list_users_validation:
        users_validation[user].append((item, rating))
        items[item].append((user, rating))
    elif user in list_users_test:
        users_test[user].append((item, rating))
        items[item].append((user, rating))

pickle.dump(users_train,        open("users_train_set.pickle", "wb"),       protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(users_validation,   open("users_validation_set.pickle", "wb"),  protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(users_test,         open("users_test_set.pickele", "wb"),        protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(items,              open("items_set.pickle", "wb"),             protocol=pickle.HIGHEST_PROTOCOL)

# data = pd.DataFrame({
#     "users":users,
#     "items":items,
#     "ratings":ratings
#     })
# data.to_csv("data.csv", sep=",")

