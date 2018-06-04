from Adaboost import adaboost
from HypothesisProvider import HypothesisProvider
import pandas
import numpy


def get_f1_score(truth, decision):

    true_positive = 0
    false_positive = 0  # type 1 error
    false_negative = 0  # type 2 error
    true_negative = 0

    n = len(truth)

    for i in range(n):
        if truth[i] == "yes":
            if decision[i] == "yes":
                true_positive += 1
            elif decision[i] == "no":
                false_negative += 1
            else:
                raise Exception("Inside F1 score: 1")
        elif truth[i] == "no":
            if decision[i] == "yes":
                false_positive += 1
            elif decision[i] == "no":
                true_negative += 1
            else:
                Exception("Inside F1 score: 2")
        else:
            Exception("Inside F1 score: 3")

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2.0 / (1/precision + 1/recall)

    print("(Precision, Recall, F1) =(%f, %f, %f )" % (precision, recall, f1_score))
    return precision, recall, f1_score


def get_combined_decision(test_set, list_learners, list_weights):

    n = test_set.shape[0]
    res = [0 for i in range(n)]

    for i in range(len(list_learners)):
        ans = list_learners[i].get_decision(test_set)
        for j in range(n):
            if ans[j] == "yes":
                res[j] += list_weights[i]
            elif ans[j] == "no":
                res[j] -= list_weights[i]
            else:
                raise RuntimeError("Inside score function")
    result = []
    for a in res:
        result.append("yes" if a > 0.0 else "no")

    return result


def driver(k_fold_splits, adaboost_rounds):

    hp = HypothesisProvider()
    training_set = pandas.read_csv("data/dataset.csv")

    splits = numpy.array_split(training_set, k_fold_splits)
    list_f1_score = []
    for i in range(k_fold_splits):
        print("Fold %d" %(i+1))
        train = pandas.concat([training_set, splits[i]]).drop_duplicates(keep=False)
        test = splits[i]

        h_i, z_i = adaboost(train, hp, adaboost_rounds)
        decision = get_combined_decision(test_set=test, list_learners=h_i, list_weights=z_i)
        truth = test["y"].tolist()

        f_i = get_f1_score(truth=truth, decision=decision)
        list_f1_score.append(f_i[2])

    return sum(list_f1_score)/k_fold_splits


def main():
    f1_score = driver(k_fold_splits=10, adaboost_rounds=10)
    print("\n\n########################################")
    print("Average F1 score =  %f" %(f1_score))


if __name__ == "__main__":
    main()