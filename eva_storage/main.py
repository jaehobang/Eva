"""
This file runs the Eva Storage Pipeline
If any issues arise please contact jaeho.bang@gmail.com

@Jaeho Bang
"""



"""
Thoughts:
Steps:
1. Load the data
2. Preprocess the data
3. CAE Network
4. Clustering
5. Compression Saving
6. Indexing
7. CBIR

"""

import numpy as np
import cv2

from loaders.loader_uadetrac import LoaderUADetrac
from eva_storage.preprocessingModule import PreprocessingModule
from eva_storage.UNet import UNet
from eva_storage.clusterModule import ClusterModule

class Runner:


    def __init__(self):
        self.loader = LoaderUADetrac()
        self.preprocess = PreprocessingModule()
        self.network = UNet()
        self.cluster = ClusterModule()


    def run(self):
        """
        Steps:
        1. Load the data
        2. Preprocess the data
        3. Train the network
        4a. Cluster the data
        4b. Postprocess the data
        5a. Generate compressed form
        5b. Generate indexes and preform CBIR
        :return: ???
        """

        # 1. Load the image
        images = self.loader.load_images()
        boxes = self.loader.load_boxes()
        labels = self.loader.load_labels()
        video_start_indices = self.loader.get_video_start_indices()

        # 2. Begin preprocessing
        segmented_images = self.preprocess.run(images, video_start_indices)

        self.network.train(images, segmented_images)
        final_compressed_images, final_segmented_images = self.network.execute()
        cluster_labels = self.cluster.run(final_compressed_images)




if __name__ == "__main__":




    runner = Runner()
    runner.run()

