# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 20:32:16 2018
@author: Tao Lin

Will make some modifications to this code for wnet replication

Training and Predicting with the W-Net unsupervised segmentation architecture
"""

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import config
import time

import eva_storage.external.wnet.tao_wnet.WNet as WNet

from loaders.uadetrac_loader import LoaderUADetrac

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation with WNet')
parser.add_argument('--in_Chans', metavar='C', default=3, type=int, 
                    help='number of input channels')
parser.add_argument('--squeeze', metavar='K', default=4, type=int, 
                    help='Depth of squeeze layer')
parser.add_argument('--out_Chans', metavar='O', default=3, type=int, 
                    help='Output Channels')
parser.add_argument('--input_folder', metavar='f', default=None, type=str, 
                    help='Folder of input images')
parser.add_argument('--output_folder', metavar='of', default=None, type=str, 
                    help='folder of output images')

args = parser.parse_args()


vertical_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,  0,  -1], 
                                            [1,  0,  -1], 
                                            [1,  0,  -1]]]])).float().to(config.device), requires_grad=False)

horizontal_sobel=torch.nn.Parameter(torch.from_numpy(np.array([[[[1,   1,  1], 
                                              [0,   0,  0], 
                                              [-1 ,-1, -1]]]])).float().to(config.device), requires_grad=False)

def gradient_regularization(softmax):
    vert=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), vertical_sobel) for i in range(softmax.shape[0])], 1)
    hori=torch.cat([F.conv2d(softmax[:, i].unsqueeze(1), horizontal_sobel) for i in range(softmax.shape[0])], 1)
    print('vert', torch.sum(vert))
    print('hori', torch.sum(hori))
    mag=torch.pow(torch.pow(vert, 2)+torch.pow(hori, 2), 0.5)
    mean=torch.mean(mag)
    return mean

def train_op(model, optimizer, input, psi=0.5):
    enc = model(input, returns='enc')
    n_cut_loss=gradient_regularization(enc)*psi
    n_cut_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    dec = model(input, returns='dec')
    rec_loss=torch.mean(torch.pow(torch.pow(input, 2) + torch.pow(dec, 2), 0.5))*(1-psi)
    rec_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return n_cut_loss, rec_loss

def test():
    wnet= WNet.WNet(4)
    wnet=wnet.to(config.device)
    synthetic_data=torch.rand((5, 3, 128, 128)).to(config.device)
    optimizer=torch.optim.SGD(wnet.parameters(), 0.001)
    train_op(wnet, optimizer, synthetic_data)

def run():
    """
    TODO:
    1. Load UA-detrac
    2. run the training code that is located in jupyter file
    loop will look like test() function in this file
    3. Save the output images on disk so that it can be loaded when doing evaluation
    4. But as a safety measure, let's visualize the images....
    :return:
    """
    batch_size = 10
    loader = LoaderUADetrac()
    images = loader.load_cached_images()
    X = images.astype(np.float)
    train_data = torch.from_numpy(X).float()
    num_epochs = 200



    N, H, W, C = images.shape
    train_data = train_data.permute(0, 3, 1, 2)
    assert (train_data.size(1) == 5)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               shuffle=True, batch_size=batch_size,
                                               num_workers=1, drop_last=True)

    model = WNet.WNet(4)
    model = model.to(config.device)
    optimizer = torch.optim.SGD(model.parameters(), 0.001)


    # configure the training loop
    print("Training starting....")

    for epoch in range(num_epochs):
        st = time.perf_counter()
        for i, images in enumerate(train_loader):
            optimizer.zero_grad()
            images_cuda = images.to(config.device)
            loss1, loss2 = train_op(model, optimizer, images_cuda)

        print('epoch [{}/{}], softcut loss:{:.4f}, reconstruction loss:{:.4f}, time elapsed:{:.4f} (sec)'
                  .format(epoch + 1, num_epochs, loss1.data, loss2.data, time.perf_counter() - st))




if __name__ == "__main__":
    test()


