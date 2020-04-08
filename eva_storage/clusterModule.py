
from __future__ import annotations
import time
from sklearn.cluster import AgglomerativeClustering
from logger import Logger
from abc import ABC, abstractmethod

# Assume you have compressed images: images_compressed
# original_images -> images_compressed (output of the encoder network)




class ClusterModule:

    def __init__(self):
        self.ac = None
        self.logger = Logger()

    def run(self, image_compressed, fps=20):
        """
        :param image_compressed:
        :param fps:
        :return: sampled frames, corresponding cluster numbers for each instance
        """
        self.logger.info("Cluster module starting....")
        n_samples = len(image_compressed)
        self.ac = AgglomerativeClustering(n_clusters=n_samples // fps)
        start_time = time.perf_counter()
        self.ac.fit(image_compressed)
        self.logger.info(f"Time to fit {n_samples}: {time.perf_counter() - start_time} (sec)")
        method = FirstEncounterMethod()
        self.logger.info(f"Sampling frames based on {str(method)} strategy")
        rep_indices = method.run(self.ac.labels_)

        return image_compressed[rep_indices], rep_indices, self.ac.labels_



    def plot_distribution(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(self.ac.labels_, bins=max(self.ac.labels_) + 1)
        plt.xlabel("Cluster Numbers")
        plt.ylabel("Number of datapoints")




class SamplingMethod(ABC):

    @abstractmethod
    def run(self, cluster_labels):
        pass



class FirstEncounterMethod(SamplingMethod):

    def __str__(self):
        return "First Encounter Method"

    def run(self, cluster_labels):
        label_set = set()
        indices_list = []
        for i, cluster_label in enumerate(cluster_labels):
            if cluster_label not in label_set:
                label_set.add(cluster_label)
                indices_list.append(i)

        return indices_list





if __name__ == "__main__":
    from eva_storage.UNet import CAENetwork
    from eva_storage.videoInputModule import VideoInputModule
    import os
    eva_dir = os.path.abspath('../')
    detrac_dir = os.path.join(eva_dir, 'data', 'ua_detrac', 'small-data')
    vim_ = VideoInputModule()
    vim_.convert_video(detrac_dir)
    image_table = vim_.get_image_array()
    cae = CAENetwork()
    cae.train(image_table)

    images_compressed = cae.get_compressed(image_table)

    cm = ClusterModule()
    image_labels = cm.run(images_compressed)
    cm.plot_distribution()
