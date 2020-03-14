"""
Defines utility functions that is needed for the pipeline to run

"""


import numpy as np

from logger import Logger, LoggingLevel


class Utils:

    def __init__(self):
        self.logger = Logger()

    def labels2binaryUAD(self, labels, categories = None, numerical = None):
        """

        :param labels: 2d array that contains the labels corresponding to each box
        :param categories: category of interest, we expect a list
        :param numerical: numbers of interest, we expect a list -- NOT SUPPORTED
        :return: dictionary that contains binary labels for all the categories provided
        """


        if categories is None and numerical is None:
            self.logger.error(f"Provide either category list or numerical breakdowns as list, Returning....")
            return None

        elif categories is None:
            self.logger.error(f"Numerical binary label generation is not yet supported! Returning...")
            return None

        label_dict = {}
        for category in categories:
            label_dict[category] = []

        for frame in labels:
            for category in label_dict.keys():
                if category in frame:
                    label_dict[category].append(1)
                else:
                    label_dict[category].append(0)

        ## make sure the length of each list is equal to the original
        for category in categories:
            assert(len(label_dict[category]) == len(labels))

        return label_dict

