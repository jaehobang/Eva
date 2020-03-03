"""

Wrapper file around uniform sampling

Expected input: np.array / list to perform uniform sampling on, sampling rate
Expected output: list of indices corresponding to selected indexes

"""
import numpy as np
from eva_storage.logger import Logger



class UniformSampling:



    @staticmethod
    def sample(arr, sample_rate=20):
        """

        :param arr:  np.ndarray / list to perform uniform sampling on
        :param sample_rate: sampling rate
        :return: sampled elements
        """
        logger = Logger()

        ## perform some sanity check
        if sample_rate <= 0:
            logger.error(f"sample rate:{sample_rate} <= 0")
        if (type(arr) is not list) and (type(arr) is not np.ndarray):
            logger.error(f"argument list type {type(arr)} but must be list or np.ndarray")


        end_point = len(arr)
        indexes = [i for i in range(0, end_point, sample_rate)]
        if type(arr) == np.ndarray:
            return arr[indexes]
        else:
            new_list = []
            for index in indexes:
                new_list.append(arr[index])
            return new_list


