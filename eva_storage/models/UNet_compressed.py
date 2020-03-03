"""
This file defines the actual torch model that we use for image compression and indexing

@Jaeho Bang
"""




import torch
import torch.nn as nn


class UNet_compressed(nn.Module):
    def __init__(self, middle_neural_count):
        super(UNet_compressed, self).__init__()

        self.K = 10
        self.output_channels = 1
        self.input_channels = 3
        self.seg_channels = 2
        self.middle_neural_count = middle_neural_count

        self.create_enc()
        input_shape = (24, 75, 75)  # input (3,40,80)
        self.create_central(input_shape)
        self.create_decoder_b()

    def forward(self, x):
        x1 = self.u_enc1(x)
        x2 = self.u_enc2(x1)

        # print(x1.shape)
        n, c, h, w = x2.shape
        compressed = self.central_layers1(x2.view(n, c * h * w))
        extended = self.central_layers2(compressed)

        xb1 = self.u_decb1(torch.cat((x2, extended.view(n, c, h, w)), dim=1))
        final_regen = self.u_decb2(torch.cat((x1, xb1), dim=1))

        # return [compressed, background_regen, foreground_regen, final_regen]
        return [compressed, final_regen]

    def create_enc(self):
        self.u_enc1 = nn.Sequential()
        self.u_enc1.add_module('Conv1_1', nn.Conv2d(self.input_channels, 8, kernel_size=3, padding=(1, 1)))
        self.u_enc1.add_module('Relu1_2', nn.ReLU(True))
        self.u_enc1.add_module('Conv1_3', nn.Conv2d(8, 8, kernel_size=3, padding=(1, 1)))
        self.u_enc1.add_module('Drop1_35', nn.Dropout2d(p=0.1))
        self.u_enc1.add_module('Relu1_4', nn.ReLU(True))

        self.u_enc2 = nn.Sequential()
        self.u_enc2.add_module('Max2_0', nn.MaxPool2d(2, stride=2))
        self.u_enc2.add_module('Conv2_1', nn.Conv2d(8, 16, kernel_size=3, padding=(1, 1)))
        self.u_enc2.add_module('Relu2_2', nn.ReLU(True))
        self.u_enc2.add_module('Conv2_3', nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1)))
        self.u_enc2.add_module('Drop2_35', nn.Dropout2d(p=0.1))
        self.u_enc2.add_module('Relu2_4', nn.ReLU(True))


    def create_central(self, input_shape):
        c, h, w = input_shape
        self.central_layers1 = nn.Sequential()
        self.central_layers1.add_module('Mlp', nn.Linear(c * h * w, self.middle_neural_count))

        self.central_layers2 = nn.Sequential()
        self.central_layers2.add_module('Mlp', nn.Linear(self.middle_neural_count, c * h * w))

    def create_decoder_b(self):

        self.u_decb2 = nn.Sequential()
        self.u_decb2.add_module('Conv5_2', nn.Conv2d(32, 16, kernel_size=3, padding=(1, 1)))
        self.u_decb2.add_module('Relu5_3', nn.ReLU(True))
        self.u_decb2.add_module('Conv5_4', nn.Conv2d(16, 8, kernel_size=3, padding=(1, 1)))
        self.u_decb2.add_module('Drop5_35', nn.Dropout2d(p=0.1))
        self.u_decb2.add_module('Relu5_5', nn.ReLU(True))
        self.u_decb2.add_module('CT5_6', nn.ConvTranspose2d(8, 8, kernel_size=2, stride=2))

        self.u_decb3 = nn.Sequential()
        self.u_decb3.add_module('Conv6_2', nn.Conv2d(16, 8, kernel_size=3, padding=(1, 1)))
        self.u_decb3.add_module('Relu6_3', nn.ReLU(True))
        self.u_decb3.add_module('Conv6_4', nn.Conv2d(8, self.output_channels, kernel_size=3, padding=(1, 1)))
        self.u_decb3.add_module('Relu6_5', nn.ReLU(True))

