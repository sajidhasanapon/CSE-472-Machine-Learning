class Hypothesis:

    def __init__(self):
        pass

    def train(self, training_set):
        """
        :param pandas.DataFrame training_set: The training set
        :return:
        """
        NotImplementedError("learn() not implemented in class: Hypothesis")

    def get_decision(self, sample):
        NotImplementedError("get_decision() not implemented in class: Hypothesis")

    #def get_info_gain(self):
     #   pass

