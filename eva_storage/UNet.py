"""
Defines the model used for generating index and compression

@Jaeho Bang
"""

import time
import os
import numpy as np
from eva_storage.models.UNet_final import UNet_final
from eva_storage.models.UNet_compressed import UNet_compressed
from eva_storage.logger import Logger, LoggingLevel


import torch
import torch.utils.data
import torch.nn as nn
import argparse
import config


parser = argparse.ArgumentParser(description='Define arguments for loader')
parser.add_argument('--learning_rate', type = int, default=0.0001, help='Learning rate for UNet')
parser.add_argument('--total_epochs', type = int, default=60, help='Number of epoch for training')
parser.add_argument('--l2_reg', type = int, default=1e-6, help='Regularization constaint for training')
parser.add_argument('--batch_size', type = int, default = 64, help='Batch size used for training')
parser.add_argument('--compressed_size', type = int, default = 100, help='Number of features the compressed image format has')
parser.add_argument('--checkpoint_name', type = str, default = 'unet_uadetrac', help='name of the file that will be used to save checkpoints')
args = parser.parse_args()


class UNet:

    def __init__(self, type = 0):
        self.model = None
        self.dataset = None
        self.data_dimensions = None
        self.logger = Logger()
        self.network_type = type ## for now, type will denoting whether we are using a compressed or full network



    def debugMode(self, mode = False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)

    def createDataExecute(self, images:np.ndarray, batch_size = args.batch_size):
        assert(images.dtype == np.uint8)
        images_normalized = np.copy(images)
        images_normalized = images_normalized.astype(np.float)

        images_normalized /= 255.0
        train_data = torch.from_numpy(images_normalized).float()
        train_data = train_data.permute(0, 3, 1, 2)

        return torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=4)


    def createData(self, images:np.ndarray, segmented_images:np.ndarray, batch_size = args.batch_size):
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
        segmented_normalized = np.copy(segmented_images)
        segmented_normalized = segmented_normalized.astype(np.float)

        images_normalized /= 255.0
        train_data = torch.from_numpy(images_normalized).float()
        train_data = train_data.permute(0,3,1,2)

        segmented_normalized /= 255.0
        seg_data = torch.from_numpy(segmented_normalized).float()
        seg_data = seg_data.unsqueeze_(-1)
        seg_data = seg_data.permute(0,3,1,2)

        return torch.utils.data.DataLoader(torch.cat((train_data, seg_data), dim = 1), batch_size = batch_size, shuffle = False, num_workers = 4)


    def _parse_dir(self, directory_string):
        """
        This function is called by other methods in UNET to parse the directory string to extract model name and epoch
        We will assume the format of the string is /dir/name/whatever/{model_name}-{epoch}.pth
        :param directory_string: string of interest
        :return:
        """

        tmp = directory_string.split('/')
        tmp = tmp[-1]
        model_name, epoch_w_pth = tmp.split('-')
        epoch = int(epoch_w_pth.split('.')[0])
        assert(type(epoch) == int)
        assert(type(model_name) == str)
        return model_name, epoch


    def train(self, images:np.ndarray, segmented_images:np.ndarray, save_name, load_dir = None):
        """
        Trains the network with given images
        :param images: original images
        :param segmented_images: tmp_data
        :return: None
        """


        self.data_dimensions = segmented_images.shape
        self.dataset = self.createData(images, segmented_images)
        model_name = save_name
        epoch = 0

        if load_dir is not None:
            self.logger.info(f"Loading from {load_dir}")
            self._load(load_dir)
            model_name, epoch = self._parse_dir(load_dir)

        if self.model is None:
            ## load_dir might not have been specified, or load_dir is incorrect
            self.logger.info(f"New model instance created on device {config.train_device}")
            if self.network_type == 0:
                self.model = UNet_final(args.compressed_size).to(device = config.train_device, dtype = None, non_blocking = False)
            elif self.network_type == 1:
                self.model = UNet_compressed(args.compressed_size).to(device = config.train_device, dtype = None, non_blocking = False)


        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)
        st = time.perf_counter()


        self.logger.info("Training the network....")
        for ep in range(epoch, args.total_epochs):
            for i, images in enumerate(self.dataset):
                images = images.to(config.train_device)
                images_input = images[:,:3,:,:]
                images_output = images[:,3:,:,:]
                compressed, final = self.model(images_input)

                optimizer.zero_grad()
                loss = distance(final, images_output)
                loss.backward()
                optimizer.step()


            self.logger.info('epoch [{}/{}], loss:{:.4f}, time elapsed:{:.4f} (sec)'.format(ep, args.total_epochs,
                                                                                       loss.data,
                                                                                       time.perf_counter() - st))
            st = time.perf_counter()

            if ep % 30 == 0:
                self._save(save_name, ep)

        self._save(save_name, args.total_epochs)
        self.logger.info(f"Finished training the network and save as {save_name+'-epoch'+str(args.total_epochs)+'.pth'}")
        return None


    def _save(self, save_name, epoch = 0):
        """
        Save the model
        We will save this in the
        :return: None
        """
        eva_dir = config.eva_dir
        dir = os.path.join(eva_dir, 'data', 'models', '{}-epoch{}.pth'.format(save_name, epoch))
        print("Saving the trained model as....", dir)

        torch.save(self.model.state_dict(), dir)


    def _load(self, load_dir, execute = False):
        """
        Load the model

        :return:
        """

        if os.path.exists(load_dir): ## os.path.exists works on folders and files

            if execute:
                if self.network_type == 0:
                    self.model = UNet_final(args.compressed_size).to(config.eval_device, dtype=None, non_blocking=False)
                elif self.network_type == 1:
                    self.model = UNet_compressed(args.compressed_size).to(config.eval_device, dtype=None,
                                                                     non_blocking=False)
            else:
                if self.network_type == 0:
                    self.model = UNet_final(args.compressed_size).to(config.eval_device, dtype=None, non_blocking=False)
                if self.network_type == 0:
                    self.model = UNet_final(args.compressed_size).to(config.eval_device, dtype=None, non_blocking=False)
                self.model = UNet_final(args.compressed_size).to(config.train_device, dtype=None, non_blocking=False)

            self.model.load_state_dict(torch.load(load_dir))
            self.logger.info("Model load success!")

        else:
            self.logger.error("Checkpoint does not exist returning")


    def execute(self, images:np.ndarray = None, load_dir = None):
        """
        We will overload this function to take in no parameters when we are just executing on the given image..
        :return: compressed, segmented images that are output of the network
        """

        st = time.perf_counter()

        if load_dir is not None:
            self.logger.info(f"Loading from {load_dir}")
            self._load(load_dir, execute=True)

        if self.model is None:
            self.logger.error("There is no model and loading directory is not supplied. Value Error will be raised")
            raise ValueError


        assert(self.model is not None)
        #self.logger.debug(f"Model on gpu device {self.model.get_device()}, running execution on gpu device {config.eval_device}")

        if images is None:
            self.logger.info("Images are not given, assuming we already have dataset object...")

            seg_data = np.ndarray(shape=self.data_dimensions)
            compressed_data = np.ndarray(shape = (self.data_dimensions[0], args.compressed_size))
            for i, images_ in enumerate(self.dataset):
                images_ = images_.to(config.eval_device)
                images_input = images_[:,:3,:,:]
                compressed, final = self.model(images_input)
                final_cpu = self._convertSegmented(final)
                compressed_cpu = self._convertCompressed(compressed)
                seg_data[i * args.batch_size:(i + 1) * args.batch_size] = final_cpu
                compressed_data[i*args.batch_size:(i + 1) * args.batch_size] = compressed_cpu
        else:
            self.logger.info("Images are given, creating dataset object and executing...    ")
            dataset = self.createDataExecute(images)
            seg_data = np.ndarray(shape=(images.shape[0], images.shape[1], images.shape[2]))
            compressed_data = np.ndarray(shape=(images.shape[0], args.compressed_size))
            self.logger.debug(f"Seg data projected shape {seg_data.shape}")
            self.logger.debug(f"Compressed data projected shape {compressed_data.shape}")
            for i, images_ in enumerate(dataset):
                images_ = images_.to(config.eval_device)
                compressed, final = self.model(images_)
                final_cpu = self._convertSegmented(final)
                compressed_cpu = self._convertCompressed(compressed)
                seg_data[i * args.batch_size:(i + 1) * args.batch_size] = final_cpu
                compressed_data[i * args.batch_size:(i + 1) * args.batch_size] = compressed_cpu


        self.logger.info(f"Processed {len(images)} in {time.perf_counter() - st} (sec)")

        return compressed_data.astype(np.uint8), seg_data.astype(np.uint8)


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





