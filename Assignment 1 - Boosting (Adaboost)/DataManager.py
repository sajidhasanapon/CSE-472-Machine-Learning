import pandas as pd
from sklearn.utils import shuffle
'''
Separating y = yes/no samples
because y = yes samples are rare
(only 4640 out of 41188)
'''
df = pd.read_csv("data/raw_data.csv")  # raw data
df_yes = df.loc[df["y"] == "yes"]      # y = yes
df_no = df.loc[df["y"] == "no"]        # y = no

# truncating df_no
df_no = df_no.sample(df_yes.shape[0])  # equal number of yes/no

#print(df_yes.shape)
#print(df_no.shape)

df_final = shuffle(pd.concat([df_yes, df_no]))    # this will be our working dataset
df_final.to_csv("data/dataset.csv", sep=",")

























# Splitting raw data into train data and test data
# This is needed for holdout test set
# Not needed in this assignment
'''
df_train, df_test = train_test_split(df, test_size=0.2) # 20% test
df_train.to_csv("data/train.csv", sep=",")
df_train.to_csv("data/test.csv", sep=",")
'''



'''
#print(df.keys()[20])
x = df.columns.get_loc("marital")
#print(x)

i = 0
for row in df.itertuples():
   print(row)
   i += 1
   if i > 10:
       break

print("###################################")

for i in range(0, 10):
    r = df.sample()
    print(r)
    print(type(r))
    #print(r["job"])

print(type(df))
print(df_train["age"].tolist())
'''