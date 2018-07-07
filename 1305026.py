import numpy as np
import pandas as pd
from train import *

X_train = pd.read_excel("ratings_train.xlsx", header=None).as_matrix()
train(ratings_train=X_train, n_factors=10, lambd=0.1)