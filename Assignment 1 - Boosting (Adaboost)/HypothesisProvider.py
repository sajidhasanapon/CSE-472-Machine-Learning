from DecisionStump import DecisionStump


class HypothesisProvider:
    def __init__(self):
        self.stump_names = ["age", "job", "marital", "education",
                            "default", "housing", "loan", "contact",
                            "day_of_week", "duration",
                            "campaign", "pdays", "previous", "poutcome",
                            "emp.var.rate", "cons.price.idx",
                            "cons.conf.idx", "euribor3m", "nr.employed"]
        self.stumps = []
        self.max_gain = -float("inf")  # negative infinity
        self.best_stump = None

    def get_hypothesis(self, sampled_train_data):
        """
        :param pandas.DataFrame sampled_train_data:
        :return: self.best_stump
        :type: DecisionStump
        """
        self.stumps = []
        self.max_gain = -float("inf")  # negative infinity
        self.best_stump = None  # type: DecisionStump

        for name in self.stump_names:
            stump_temp = DecisionStump(name)
            stump_temp.train(sampled_train_data)
            self.stumps.append(stump_temp)

        for stump in self.stumps:
            gain_temp = stump.get_info_gain()
            if gain_temp > self.max_gain:
                self.max_gain = gain_temp
                self.best_stump = stump

        return self.best_stump
