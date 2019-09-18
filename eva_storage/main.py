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
from eva_storage.indexingModule import IndexingModule

class Runner:


    def __init__(self):
        self.loader = LoaderUADetrac()
        self.preprocess = PreprocessingModule()
        self.network = UNet()
        self.cluster = ClusterModule()
        self.index = IndexingModule()


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
        import time
        st = time.time()
        # 1. Load the image
        images = self.loader.load_cached_images()
        labels = self.loader.load_cached_labels()
        vehicle_labels = labels['vehicle']
        video_start_indices = self.loader.get_video_start_indices()
        print("Done loading images in", time.time() - st, "(sec)")

        # 2. Begin preprocessing
        st = time.time()
        segmented_images = self.preprocess.run(images, video_start_indices)
        print("Done with background subtraction in", time.time() - st, "(sec)")

        st = time.time()
        self.network.train(images, segmented_images)
        final_compressed_images, final_segmented_images = self.network.execute()
        print("Done training the main network in", time.time() - st, "(sec)")

        st = time.time()
        cluster_labels = self.cluster.run(final_compressed_images)
        print("Done clustering in", time.time() - st, "(sec)")

        st = time.time()
        self.index.train(images, final_segmented_images, vehicle_labels)




if __name__ == "__main__":




    runner = Runner()
    runner.run()

