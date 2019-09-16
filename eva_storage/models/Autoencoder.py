"""
Autoencoder file for building index

@Jaeho Bang
"""

"""
This file defines the actual torch model that we use for image compression and indexing

@Jaeho Bang
"""




import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(3, 6, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 6, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(6, 9, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(9, 9, kernel_size=2, stride=2),
            nn.Conv2d(9, 9, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(9, 6, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.ConvTranspose2d(6, 6, kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(6, 6, kernel_size=3, padding=(1, 1)),
            nn.ReLU(True),
            nn.Conv2d(6, 3, kernel_size=3, padding=(1, 1)),
            nn.Sigmoid())

    def forward(self, x):
        compressed = self.encoder(x)
        x = self.decoder(compressed)
        return compressed, x
