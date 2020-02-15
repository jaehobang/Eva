import time
from sklearn.cluster import AgglomerativeClustering


# Assume you have compressed images: images_compressed
# original_images -> images_compressed (output of the encoder network)

class ClusterModule:

    def __init__(self):
        self.ac = None

    def run(self, image_compressed, fps=20):
        n_samples = len(image_compressed)
        self.ac = AgglomerativeClustering(n_clusters=n_samples // fps)
        start_time = time.time()
        self.ac.fit(image_compressed)
        print("Time to fit ", n_samples, ": ", time.time() - start_time)
        return self.ac.labels_

    def plot_distribution(self):
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
        axs.hist(self.ac.labels_, bins=max(self.ac.labels_) + 1)
        plt.xlabel("Cluster Numbers")
        plt.ylabel("Number of datapoints")



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
