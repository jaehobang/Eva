"""
This file replicates WNet by https://arxiv.org/pdf/1711.08506.pdf

@Jaeho Bang
"""

import numpy as np
import time
import os
import cv2
import sys

import torch
import torch.utils.data
import torch.nn as nn

import config


from loaders.loader_uadetrac import LoaderUADetrac

class WNet:
    def __init__(self):
        num_epochs = 100
        batch_size = 16



if __name__ == "__main__":
    loader = LoaderUADetrac()
    images = loader.load_cached_images()









