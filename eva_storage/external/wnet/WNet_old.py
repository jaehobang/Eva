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

from eva_storage.external.wnet.WNet_model import WNet_model
from loaders.uadetrac_loader import LoaderUADetrac




class WNet:
    def __init__(self):
        self.num_epochs = 100
        self.batch_size = 10
        self.model = WNet_model()
        self.model.to(config.device)


    def train(self, train_loader):
        learning_rate = 0.0001
        optimizer = torch.optim.Adam(self.model.parameters(), lr = learning_rate)
        distance = nn.MSELoss()
        print("Training starting....")
        for epoch in range(self.num_epochs):
            st = time.perf_counter()
            for i, images in enumerate(train_loader):
                optimizer.zero_grad()
                images_cuda = images.to(config.device)
                N,C,H,W = images_cuda.size()
                dist_diff_matrix = self.calculate_dist_matrix(N,H,W)
                segmented, recon = self.model(images_cuda)
                loss1 = self.softcut_loss(segmented, images_cuda, dist_diff_matrix)
                loss1.backward(retrain_graph=True)
                optimizer.step()
                optimizer.zero_grad()
                loss2 = distance(recon, images_cuda)
                loss2.backward(retrain_graph = False)
                optimizer.step()
                print('epoch [{}/{}], loss2:{:.4f}, time elapsed:{:.4f} (sec)'
                      .format(epoch + 1, self.num_epochs, loss2.data, time.perf_counter() - st))


    def calculate_denom(self, segmented_imgs, weight_matrix):
        # segmented_imgs: torch.Size(N,K,H,W)
        # weight_matrix: torch.Size(N,H,W,HxW)
        # steps:
        # 1. create the expanded matrix
        # 2. multiply with weight_matrix
        # 3. tensor sum wrt depth
        # 4. tensor sum wrt h,w
        # 5. should be a number
        # seg = segmented_imgs
        # seg = torch.tensor(segmented_imgs.data.clone(), requires_grad = True)
        """
        seg = segmented_imgs.clone()
        N, K, H, W = seg.size()
        seg.unsqueeze_(-1)
        seg = seg.expand(-1, -1, -1, -1, H * W)
        assert (seg.size() == weight_matrix.size())
        seg = torch.mul(seg, weight_matrix)
        return seg.sum((2, 3, 4))
        """
        ## I want to try this without using clone
        N,K,H,W = segmented_imgs.size()
        seg = segmented_imgs.unsqueeze(-1).expand(-1, -1, -1, -1, H*W)
        assert(seg.size() == weight_matrix.size())
        seg *= weight_matrix
        return seg.sum((2,3,4))


    def calculate_num(self, segmented_imgs, weight_matrix):
        # segmented_imgs: torch.Size(N,K,H,W)
        # weight_matrix will be of size (N,H,W,HxW)
        """
        N, K, H, W = segmented_imgs.size()
        # seg1 = torch.tensor(segmented_imgs.data.clone(), requires_grad = True)
        # seg2 = torch.tensor(seg1.clone(), requires_grad = True)
        seg1 = segmented_imgs.clone()
        seg2 = segmented_imgs.clone()

        seg1.unsqueeze_(-1)  # N,K,H,W,1
        seg1 = seg1.expand(-1, -1, -1, -1, H * W)
        seg2.unsqueeze_(-1)  # N,K,H,W,1
        seg2 = seg2.reshape(N, K, 1, 1, -1)  # N,K,1,1,H*W
        seg2 = seg2.expand(-1, -1, H, W, -1)  # N,K,H,W,H*W
        assert (seg1.size() == seg2.size())
        seg1 = torch.mul(seg1, seg2)  # N,K,H,W,H*W

        seg1 = torch.mul(seg1, weight_matrix)  # N,K,H,W,H*W
        return seg1.sum((2, 3, 4))
        """
        N,K,H,W = segmented_imgs.size()
        seg1 = segmented_imgs.unsqueeze(-1).expand(-1,-1,-1,-1, H*W)
        seg2 = segmented_imgs.unsqueeze(-1).reshape(N,K,1,1,-1).expand(-1,-1,H,W,-1)
        assert(seg1.size() == seg2.size())
        seg1 = seg1 * seg2 * weight_matrix
        return seg1.sum((2,3,4))



    def calculate_dist_matrix(self, x_positional_matrix, y_positional_matrix):
        sigma_x_squared = 16
        radius_threshold = 5
        # x_matrix1 = torch.arange(end = H, dtype=torch.float, requires_grad=True).cuda()
        x_matrix1 = torch.arange(end=H, dtype=torch.float).to(config.device)

        x_matrix1.unsqueeze_(-1)
        x_matrix1 = x_matrix1.expand(-1, W)
        x_matrix1.unsqueeze_(-1)
        x_matrix1 = x_matrix1.expand(-1, -1, H * W)

        # x_matrix2 = torch.arange(end = H, dtype = torch.float, requires_grad = True).cuda()
        x_matrix2 = torch.arange(end=H, dtype=torch.float).to(config.device)
        x_matrix2.unsqueeze_(-1)
        x_matrix2 = x_matrix2.expand(-1, W)
        x_matrix2.unsqueeze_(-1)
        x_matrix2 = x_matrix2.reshape(1, 1, -1)
        x_matrix2 = x_matrix2.expand(H, W, -1)

        x_matrix1 = x_matrix1 - x_matrix2
        x_matrix1 = torch.pow(x_matrix1, 2)

        # y_matrix1 = torch.arange(end = W, dtype=torch.float, requires_grad=True ).cuda()
        y_matrix1 = torch.arange(end=W, dtype=torch.float).to(config.device)
        y_matrix1.unsqueeze_(-1)
        y_matrix1 = y_matrix1.expand(-1, H)
        y_matrix1 = y_matrix1.permute(1, 0)
        y_matrix1.unsqueeze_(-1)
        y_matrix1 = y_matrix1.expand(-1, -1, H * W)

        # y_matrix2 = torch.arange(end = W, dtype = torch.float, requires_grad = True).cuda()
        y_matrix2 = torch.arange(end=W, dtype=torch.float).to(config.device)
        y_matrix2 = y_matrix2.repeat(H)
        y_matrix2.unsqueeze_(-1)
        y_matrix2.unsqueeze_(-1)
        y_matrix2 = y_matrix2.permute(1, 2, 0)
        y_matrix2 = y_matrix2.expand(H, W, -1)

        y_matrix1 = y_matrix1 - y_matrix2
        y_matrix1 = torch.pow(y_matrix1, 2)
        dist_threshold_matrix = torch.sqrt(x_matrix1 + y_matrix1)

        dist_diff = torch.exp(-torch.div(x_matrix1 + y_matrix1, sigma_x_squared))
        assert (dist_diff.size() == torch.Size([H, W, H * W]))
        dist_diff.unsqueeze_(0)
        dist_diff = dist_diff.expand(N, -1, -1, -1)  # N,H,W,H*W
        dist_threshold_matrix.unsqueeze_(0)
        dist_threshold_matrix = dist_threshold_matrix.expand(N,-1,-1,-1)
        assert(dist_diff.size() == dist_threshold_matrix.size())
        dist_diff[dist_threshold_matrix >= radius_threshold] = 0

        ## we need to add in the condition that if the distance is greater than threshold, it becomes 0

        return dist_diff

    def create_weight_matrix(self, original_img, dist_diff, K):
        # given the corresponding height and width, will create a weight matrix of size H,W,HxW according to the formula given in
        # louis_wnet paper.
        # original_img size: torch.Size(N,C,H,W)
        # dist_diff size: torch.Size(N,H,W,H*W)
        # we will assume matrix of size H, W is given, it is initialized to zeros
        sigma_i_squared = 100
        N,C,H,W = original_img.size()

        # matrix1 = original_img
        # matrix1 = torch.tensor(original_img.clone(), requires_grad = True)
        matrix1 = original_img.clone().to(config.device)
        matrix1.unsqueeze_(-1)
        matrix1 = matrix1.expand(-1, -1, -1, -1, H * W)

        # matrix2 = torch.tensor(original_img.clone(), requires_grad = True)
        # matrix2 = original_img
        matrix2 = original_img.clone().to(config.device)
        matrix2.unsqueeze_(-1)
        matrix2 = matrix2.reshape(N, C, 1, 1, -1).expand(-1, -1, H, W, -1)  # N,C,H,W,H*W
        assert (matrix2.size() == torch.Size([N, C, H, W, H * W]))

        matrix1 = matrix1 - matrix2
        matrix1 = torch.pow(matrix1, 2)  # N,C,H,W,H*W

        weight = torch.exp(-torch.div(matrix1.sum(1), sigma_i_squared))  # N,H,W,H*W
        assert (weight.size() == dist_diff.size())

        weight *= dist_diff
        weight.unsqueeze_(1)
        weight = weight.expand(-1, K, -1, -1, -1)  # N,K,H,W,H*W

        return weight


    def append_positional_layers(self, original_layers, H,W):
        x_position_matrix = torch.arange(end=H, dtype=torch.float)
        x_position_matrix = x_position_matrix.unsqueeze(-1).expand(-1, W)

        y_position_matrix = torch.arange(end=W, dtype=torch.float)
        y_position_matrix = y_position_matrix.unsqueeze(-1).expand(-1, H).permute(1, 0)
        return torch.cat([original_layers, x_position_matrix, y_position_matrix])


    # TODO: Need to convert this to 4d matrix because we expect to pass in multiple samples
    def softcut_loss(self, segmented_imgs, original_imgs, dist_diff_matrix):
        N, K, H, W = segmented_imgs.size()

        weight_matrix = self.create_weight_matrix(original_imgs, dist_diff_matrix, K)

        num = self.calculate_num(segmented_imgs, weight_matrix)
        denom = self.calculate_denom(segmented_imgs, weight_matrix)

        # returned num, denom is torch.Size([N,K])

        total_sum = torch.sum(torch.div(num, denom))
        return N * K - total_sum

    def calculate_dist_matrix(self, positional_matrix):
        ## positional_matrix shape > N, C, H, W
        sigma_x_squared = 16
        radius_threshold = 5
        # x_matrix1 = torch.arange(end = H, dtype=torch.float, requires_grad=True).cuda()
        N, C, H, W = positional_matrix.size()

        matrix = positional_matrix[0].permute(1,2,0) # we take a slice of the original positional_matrix
        matrix_reshaped = matrix.reshape(H*W, C)
        dists = torch.norm(matrix_reshaped[:,None] - matrix_reshaped, dim=2, p=2)
        assert(dists.size() == torch.Size([H*W, H*W]))
        dists.unsqueeze(-1)
        dists = dists.reshape(H, W, H*W) # this is normalized thus not squared
        dist_diffs = torch.exp(-torch.div(torch.pow(dists, 2), sigma_x_squared))
        dist_diffs[dists >= radius_threshold] = 0
        dist_diffs = dist_diffs.unsqueeze(0).expand(N, -1, -1, -1)


        ## we need to add in the condition that if the distance is greater than threshold, it becomes 0

        return dist_diffs



if __name__ == "__main__":
    batch_size = 64
    loader = LoaderUADetrac()
    images = loader.load_cached_images()
    X_norm = images.astype(np.float) / 255.0
    train_data = torch.from_numpy(X_norm).float()
    N,H,W,C = images.shape
    model = WNet()
    train_data = model.append_positional_layers(train_data, H,W)

    train_data = train_data.permute(0,3,1,2)
    assert(train_data.size(1) == 5)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               shuffle=True, batch_size = batch_size,
                                               num_workers = 4, drop_last = True)


    model.train(train_loader)












