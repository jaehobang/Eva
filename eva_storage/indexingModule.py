"""
This module performs the indexing task
Takes the output of the network as input.
Performs post-processing computer vision methods
Performs patch generation
performs CBIR (content based image retrieval)

@Jaeho Bang
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse

from eva_storage.models.Autoencoder import Autoencoder
from sklearn.neighbors import NearestNeighbors


DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

parser = argparse.ArgumentParser(description='Define arguments for loader')
parser.add_argument('--learning_rate', type = int, default=0.0001, help='Learning rate for UNet')
parser.add_argument('--total_epochs', type = int, default=30, help='Number of epoch for training')
parser.add_argument('--l2_reg', type = int, default=1e-6, help='Regularization constaint for training')
parser.add_argument('--batch_size', type = int, default = 32, help='Batch size used for training')
parser.add_argument('--neighbors', type = int, default = 5, help="Number of neighbors to consider for KNN")
args = parser.parse_args()


class IndexingModule:
    def __init__(self):
        self.patch_width = 32
        self.patch_height = 32
        self.max_patch_count = 10

        self.knn = NearestNeighbors(n_neighbors=args.neighbors)
        self.model = Autoencoder()
        self.dataloader = None



    def train(self, X_original:np.ndarray, X_segmented:np.ndarray, y:np.ndarray):
        patches, patch_count_list = self._postProcess(X_original, X_segmented)
        self._trainNetwork(patches, patch_count_list)
        self._createIndex(patches, patch_count_list)



    def _createIndex(self, patches:np.ndarray, patch_count_list:list):
        """
        creates boxes and organize them into patches
        :param seg_matrix: post processed segmented image matrix
        :return: patches
        """
        ### get compressed format. reshape it.
        ### compressed should be 9x16x16
        csize = 8
        cchannels = 9
        batch_size = 24

        patches_compressed = np.ndarray(shape=(sum(patch_count_list),
                                               csize ** 2 * cchannels))
        print(patches_compressed.shape)

        for i, data in enumerate(self.dataloader):
            img = data
            img_cuda = img.to(DEVICE)
            compressed, output = self.model(img_cuda)
            patches_compressed[(i * batch_size):(i * batch_size) + batch_size, :] = compressed.view(compressed.size(0),
                                                                                                    -1).cpu().detach().numpy()


        knn = NearestNeighbors(n_neighbors=args.neighbors).fit(patches_compressed)



        self.knn.fit(patches_compressed)
        # Apply KNN and show the retrived images from test set
        distances, indices = self.knn.kneighbors(patches_compressed)

        patches = patches.astype(np.uint8)



    def _trainNetwork(self, patches, patch_count_list):
        """
        :param patches:
        :param patch_count_list:
        :return:
        """
        patch_flattened = self._flattenPatches(patches, patch_count_list)

        self.dataloader = torch.utils.data.DataLoader(patch_flattened, batch_size=args.batch_size, shuffle=False, num_workers=4)
        self.model.to(DEVICE)
        distance = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=1e-5)
        print("Training the network for indexing...")
        for epoch in range(args.total_epochs):
            for data in self.dataloader:
                img = data
                img_cuda = img.to(DEVICE)
                # ===================forward=====================
                compressed, output = self.model(img_cuda)
                loss = distance(output, img_cuda)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, args.total_epochs, loss.data))


    def _flattenPatches(self, patches:np.ndarray, patch_count_list:np.ndarray):
        """
        Modifies the shape of the patch array so that the network can take
        :param patches:
        :param patch_count_list:
        :return:
        """
        n_samples = sum(patch_count_list)
        patches_flattened = np.ndarray(shape = (n_samples, self.patch_height, self.patch_width))
        currid = 0
        for frameid, frame_patch_count in enumerate(patch_count_list):
            for patchid in range(frame_patch_count):
                patches_flattened[currid] = patches[frameid][patchid]
                currid += 1
        return patches_flattened






    def _postProcess(self, image_matrix:np.ndarray, seg_matrix:np.ndarray):
        """
        perform post processing step (cv ops)
        :param seg_matrix: output of network (segmented images)
        :return: patch_array (np.ndarray), and box_count_list
        """
        patch_count_list = []
        n_samples = seg_matrix.shape[0]
        patch_array = np.ndarray(shape = (n_samples, self.max_patch_count, self.patch_height, self.patch_width))


        for ii in range(seg_matrix.shape[0]):

            labels, results = self.post_individual(seg_matrix[ii])
            assert (len(labels) == len(results))
            print(labels)

            post = results[-1]
            patches = self.detect_patches(post)
            new_patches = self.filter_patches(patches) #this should be coordinates not images
            if new_patches == None:
                patch_count_list.append(0)
            else:
                patch_count_list.append(min(len(new_patches), self.max_patch_count))
                for j in range(min(len(new_patches), self.max_patch_count)):
                    patch = new_patches[j]
                    ## TODO: need to make sure the new_patches that are detected are in a certain format
                    crop_img = image_matrix[ii][patch[1]:patch[1] + patch[3], patch[0]:patch[0] + patch[2]]

                    patch_resized = cv2.resize(crop_img, (self.patch_width, self.patch_height))
                    patch_array[ii][j] = patch_resized

        return patch_array, patch_count_list


    def post_individual(self, seg_img):
        """
        1. cv2.GaussianBlur()
        1. cv2.medianBlur()
        2. cv2.threshold(img, low, max, cv2.THRESH_OTSU)
        3. cv2.threshold(img, low, max, cv2.BINARY)
        4. cv2.erode(img, kernel, iterations)
        5. cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations )
        6. cv2.dilate(img, kernel, iterations)
        """
        seg_cp = np.copy(seg_img)
        med = cv2.medianBlur(seg_cp, 5)
        ret, otsu_m = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)

        labels = ['median blur', 'ostu on median']
        return (labels, [med, otsu_m])



    def detect_patches(self, img, shape='rectangle', mode=2):
        """
        Detects the patches from segmented images using contour detection
        :param img: segmented / grayscale image
        :param shape: the desired detected shape
        :param mode: algorithm used for detecting the contours
        :return: bounding boxes
        """
        if mode == 1:
            contours, aaaa = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif mode == 2:
            contours, aaaa = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            print("No contours in image")
            return None

        if shape != 'rectangle':
            print("Method not support")
            return None
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        return boundRect



    def filter_patches(self, patches, img_height=300, img_width=300,
                       min_ratio_image=0.05, max_ratio_image=0.7,
                       min_ratio_patch=0.5, max_ratio_patch=3.0):
        """
        Filters out patches that are too small or doesn't respect the aspect ratio
        :param patches: list of patch (starting col, starting row, width, height)
        :param img_height: height of image
        :param img_width: width of image
        :param min_ratio_image: ratio of patch to image (both width and height need to satisfy this constraint)
        :param max_ratio_image: ratio of patch to image (both width and height need to satisfy this constraint)
        :param min_ratio_patch: ratio of patch_height to patch_width
        :param max_ratio_patch: ratio of patch_height to patch_width
        :return: filtered patches
        """
        # we want to filter all the patches and return new patches that satisfy the constraint
        new_patches = []

        if patches == None:
            return None
        for patch in patches:

            patch_height = patch[3]
            patch_width = patch[2]
            height_ratio = patch_height / img_height
            width_ratio = patch_width / img_width
            ratio_patch = patch_height / patch_width
            if height_ratio >= min_ratio_image and height_ratio <= max_ratio_image and \
                    width_ratio >= min_ratio_image and width_ratio <= max_ratio_image and \
                    ratio_patch >= min_ratio_patch and ratio_patch <= max_ratio_patch:
                new_patches.append(patch)
        return new_patches

