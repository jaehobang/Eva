"""
Revised from [FCN code by shekkizh] (https://github.com/shekkizh/FCN.tensorflow)
"""


from loaders.uadetrac_loader import UADetracLoader
import numpy as np




def create_BatchDataset():
    loader = UADetracLoader()
    images = loader.load_images(image_size = 128)


    n_samples = len(images)
    train_images = images[:int(n_samples * 0.8)]
    val_images = images[int(n_samples * 0.8):]


    train_dataset = BatchDatasetUAD(train_images, True)
    valid_dataset = BatchDatasetUAD(val_images, False)

    return train_dataset, valid_dataset


class BatchDatasetUAD:
    images = []
    annotations = []
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, images, is_shuffle=False):
        """

        """

        print("Initializing Batch Dataset Reader...")
        self.images = images

        if is_shuffle:
            self.shuffle_data()
        return


    def shuffle_data(self):
        randperm = np.random.permutation(len(self.images))
        self.images = self.images[randperm]

    def get_records(self):
        return self.images

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > len(self.images):
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            self.shuffle_data()
            # Start next epoch
            start = 0
            self.batch_offset = batch_size
        end = self.batch_offset
        return self.images[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, len(self.images), size=[batch_size])
        return self.images[indexes]
