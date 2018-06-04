from HypothesisProvider import HypothesisProvider
from math import log2

"""
Russell and Norvig, 3rd edition
-------------------------------
Slight modification has been done to the pseudocode
(Checking error >= 0.5 ?)
"""


def adaboost(examples, hypothesis_provider, rounds):
    """
    :param  pandas.DataFrame examples:
    :param  HypothesisProvider hypothesis_provider:
    :param  int rounds:
    """

    h = []  # list of weak learners
    z = []  # weights of weak learners
    n = examples.shape[0]  # number of samples in training set
    w = [1.0/n for i in range(n)]  # weights of samples

    k = 0
    while k < rounds:
        print("Round %d" %(k+1))
        resampled = examples.sample(n=n, replace=True, weights=w)
        h_k = hypothesis_provider.get_hypothesis(resampled)  # type: DecisionStump

        error = 0
        verdict = []

        res = h_k.get_decision(resampled)
        truth = resampled["y"].tolist()
        # print(truth)

        for j in range(n):
            if res[j] != truth[j]:
                error += w[j]
            verdict.append(res[j] == truth[j])

        if error >= 0.5:
            print("Error = %f, continuing..." % (error))
            continue

        for j in range(n):
            if verdict[j] == 1:
                w[j] = w[j] * error / (1 - error)

        w = [float(i) / sum(w) for i in w]  # normalizing
        h.append(h_k)
        z.append(log2((1 - error) / error))  # weight of weak learner

        #print(h_k.get_name())
        #print(error)
        #print(z[k])

        k += 1

    return h, z
