"""
Defines the model used for generating index and compression

@Jaeho Bang
"""

import time
import numpy as np
from eva_storage.models.UNet_final import UNet_final


import torch
import torch.nn as nn
import argparse


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Define arguments for loader')
parser.add_argument('--learning_rate', type = int, default=0.0001, help='Learning rate for UNet')
parser.add_argument('--total_epochs', type = int, default=100, help='Number of epoch for training')
parser.add_argument('--l2_reg', type = int, default=1e-6, help='Regularization constaint for training')
parser.add_argument('--batch_size', type = int, default = 64, help='Batch size used for training')
parser.add_argument('--compressed_size', type = int, default = 100, help='Number of features the compressed image format has')
args = parser.parse_args()


class UNet:

    def __init__(self):
        self.model = UNet_final(args.compressed_size).to(DEVICE)
        self.dataset = None
        self.data_dimensions = None

    def createData(self, images:np.ndarray, segmented_images:np.ndarray):
        """
        Creates the dataset for training
        :param images: original, unnormalized images
        :param segmented_images: segmented images
        :return:
        """


        # we assume the data is not normalized...
        assert(images.dtype == np.uint8)
        assert(segmented_images.dtype == np.uint8)

        images_normalized = np.copy(images)
        images_normalized = images_normalized.astype(np.float)

        images_normalized /= 255.0
        train_data = torch.from_numpy(images_normalized)
        train_data = train_data.permute(0,3,1,2)

        segmented_images /= 255
        seg_data = torch.from_numpy(segmented_images).float()
        seg_data = seg_data.unsqueeze_(-1)
        seg_data = seg_data.permute(0,3,1,2)

        return torch.utils.data.DataLoader(torch.cat((train_data, seg_data), dim = 1))



    def train(self, images:np.ndarray, segmented_images:np.ndarray):
        """
        Trains the network with given images
        :param images: original images
        :param segmented_images: segmented_images
        :return: None
        """
        self.data_dimensions = segmented_images.shape
        self.dataset = self.createData(images, segmented_images)
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        st = time.perf_counter()

        for epoch in range(args.total_epochs):
            for i, images in enumerate(self.dataset):
                images = images.to(DEVICE)
                images_input = images[:,:3,:,:]
                images_output = images[:,3:,:,:]
                compressed, final = self.model(images_input)

                optimizer.zero_grad()
                loss = distance(final, images_output)
                loss.backward()
                optimizer.step()

            print('epoch [{}/{}], loss:{:.4f}, time elapsed:{:.4f} (sec)'.format(epoch + 1, args.total_epochs,
                                                                                       loss.data,
                                                                                       time.perf_counter() - st))

        return None


    def execute(self):
        """

        :return: compressed, segmented images that are output of the network
        """
        seg_data = np.ndarray(shape=self.data_dimensions)
        compressed_data = np.ndarray(shape = (self.data_dimensions[0], args.compressed_size))
        for i, images in enumerate(self.dataset):
            images = images.to(DEVICE)
            images_input = images[:,:3,:,:]
            compressed, final = self.model(images_input)
            final_cpu = self._convertSegmented(final)
            compressed_cpu = self._convertCompressed(compressed)
            seg_data[i * args.batch_size:(i + 1) * args.batch_size] = final_cpu
            compressed_data[i*args.batch_size:(i + 1) * args.batch_size] = compressed_cpu


        return compressed_data, seg_data


    def _convertCompressed(self, compressed_image):
        """
        Converts the compressed images from the network to something that can be saved in an np.array
        :param compressed_image: Compressed image from the network
        :return: np compressed image
        """
        compressed_cpu = compressed_image.detach().cpu().numpy()
        compressed_cpu *= 255
        compressed_cpu = compressed_cpu.astype(np.uint8)
        return compressed_cpu



    def _convertSegmented(self, segmented_image):
        """
        Converts the segmented images from the network to something that can be saved in an nparray
        :param segmented_image: Segmented image output from the network
        :return: np segmented image
        """
        recon_p = segmented_image.permute(0, 2, 3, 1)
        recon_imgs = recon_p.detach().cpu().numpy()
        recon_imgs *= 255
        recon_imgs = recon_imgs.astype(np.uint8)
        recon_imgs = recon_imgs.squeeze()
        return recon_imgs





