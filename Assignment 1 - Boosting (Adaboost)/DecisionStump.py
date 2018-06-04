from HypothesisSpace import Hypothesis
from math import log2
import numpy


class DecisionStump(Hypothesis):

    def __init__(self, attribute_name):
        self.count = {}
        self.decision = {}
        self.values = []
        self.attribute_name = attribute_name
        self.gain = 0
        self.index = 0
        self.numeric = False
        self.numeric_attributes = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
        self.mean = 0
        self.sd = 0

        if attribute_name in self.numeric_attributes:
            self.numeric = True
            self.values = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
            # self.values = [-1, 1]

        elif attribute_name == "job":
            self.values = ["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]

        elif attribute_name == "marital":
            self.values = ['divorced', 'married', 'single', 'unknown']

        elif attribute_name == "education":
            self.values = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']

        elif attribute_name == "default":
            self.values = ['no', 'yes', 'unknown']

        elif attribute_name == "housing":
            self.values = ['no', 'yes', 'unknown']

        elif attribute_name == "loan":
            self.values = ['no', 'yes', 'unknown']

        elif attribute_name == "contact":
            self.values = ['cellular', 'telephone']

        elif attribute_name == "month":
            self.values = ['jan', 'feb', 'mar', 'apr', 'may', 'june', 'july', 'aug', 'sep', 'oct', 'nov', 'dec']

        elif attribute_name == "day_of_week":
            self.values = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']

        elif attribute_name == "poutcome":
            self.values = ['failure', 'nonexistent', 'success']

        for value in self.values:
            self.count[value] = [0, 0]

    def train(self, training_set):
        if self.numeric:
            vals = training_set[self.attribute_name].tolist()
            self.mean = numpy.mean(vals)
            self.sd = numpy.std(vals)

            for value in self.values:
                self.count[value][0] = training_set.loc[(round((training_set[self.attribute_name] - self.mean) / self.sd) == value) & (training_set["y"] == "yes")].shape[0]
                self.count[value][1] = training_set.loc[(round((training_set[self.attribute_name] - self.mean) / self.sd) == value) & (training_set["y"] == "no")].shape[0]
                self.decision[value] = "yes" if self.count[value][0] > self.count[value][1] else "no"
        else:
            for value in self.values:
                self.count[value][0] = training_set.loc[(training_set[self.attribute_name] == value) & (training_set["y"] == "yes")].shape[0]
                self.count[value][1] = training_set.loc[(training_set[self.attribute_name] == value) & (training_set["y"] == "no")].shape[0]
                self.decision[value] = "yes" if self.count[value][0] > self.count[value][1] else "no"

        # print(self.count)

    def get_decision(self, sample):
        res = []
        values = sample[self.attribute_name].tolist()

        if self.numeric:
            for val in values:
                Z_score = round((val - self.mean) / self.sd)
                res.append(self.decision[Z_score] if Z_score in self.decision else "no")
        else:
            for val in values:
                res.append(self.decision[val] if val in self.decision else "no")  # default = random

        return res

    def get_info_gain(self):
        for value in self.values:
            yes, no = self.count[value][0], self.count[value][1]

            total = yes + no
            if total == 0: # yes = 0 and no = 0
                continue
            elif no == 0:
                self.gain += yes * log2(yes / total)
            elif yes == 0:
                self.gain += no * log2(no/total)
            else:
                self.gain += (yes * log2(yes/total) + no * log2(no/total))

        return self.gain
        # Did not divide by N because N is the same for all the stumps
        # Assumed 0 entropy at root
        # Therefore, self.gain will be < 0

    def get_name(self):
        return self.attribute_name

























"""
    def train(self, training_set):
        if self.numeric:
            vals = training_set[self.attribute_name].tolist()
            self.mean = numpy.mean(vals)
            # self.sd = numpy.std(vals)

            self.count[-1][0] = training_set.loc[((training_set[self.attribute_name] - self.mean) < 0) & (training_set["y"] == "yes")].shape[0]
            self.count[-1][1] = training_set.loc[((training_set[self.attribute_name] - self.mean) < 0) & (training_set["y"] == "no")].shape[0]
            self.decision[-1] = "yes" if self.count[-1][0] > self.count[-1][1] else "no"

            self.count[1][0] = training_set.loc[((training_set[self.attribute_name] - self.mean) > 0) & (training_set["y"] == "yes")].shape[0]
            self.count[1][1] = training_set.loc[((training_set[self.attribute_name] - self.mean) > 0) & (training_set["y"] == "no")].shape[0]
            self.decision[1] = "yes" if self.count[1][0] > self.count[1][1] else "no"
        else:
            for value in self.values:
                self.count[value][0] = training_set.loc[(training_set[self.attribute_name] == value) & (training_set["y"] == "yes")].shape[0]
                self.count[value][1] = training_set.loc[(training_set[self.attribute_name] == value) & (training_set["y"] == "no")].shape[0]
                self.decision[value] = "yes" if self.count[value][0] > self.count[value][1] else "no"

    def get_decision(self, sample):
        res = []
        values = sample[self.attribute_name].tolist()

        if self.numeric:
            for val in values:
                res.append(self.decision[-1] if val<0.0 else self.decision[1])
        else:
            for val in values:
                res.append(self.decision[val] if val in self.decision else "no")  # default = random

        return res
"""
