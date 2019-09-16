"""
This module performs the indexing task
Takes the output of the network as input.
Performs post-processing computer vision methods
Performs patch generation
Organizes in a format that is easily used by the CBIR module

@Jaeho Bang
"""

import numpy as np
import cv2






def postProcess(self, seg_matrix):
    ## Assume seg_matrix is of shape (n_samples, height, width)
    seg_matrix = seg_matrix.astype(np.uint8)
    seg_post = np.ndarray(shape=seg_matrix.shape)
    n_samples, height, width = seg_matrix.shape
    # We need to reshape the image 2 times because of dilation and erosion processes

    for i in range(n_samples):
        curr = np.copy(seg_matrix[i])
        # 1. threshold the image
        ret, thresh = cv2.threshold(curr, 0, 255, cv2.THRESH_OTSU)
        # 2. erode the image
        kernel = np.ones((4, 4), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
        # 3. dilate the image
        seg_post[i] = cv2.dilate(opening, kernel, iterations=3)

    return seg_post