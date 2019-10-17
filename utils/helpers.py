"""
Implementation of helper functions through the system

@Jaeho Bang
"""

import numpy as np


def generateBinaryLabels(y:list, label_of_interest = 'car'):
    """

    :param y: output from loader load labels (uadetrac)
    :return: np.array of binary labels
    """

    new_arr = np.zeros(shape = (len(y)))
    for i in range(len(y)):
        if label_of_interest in y[i]:
            new_arr[i] = 1

    return new_arr


