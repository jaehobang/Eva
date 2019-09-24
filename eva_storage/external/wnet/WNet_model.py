"""
Defines the architecture of WNet

@Jaeho Bang
"""
import torch
import torch.nn as nn


# Writing our model
class WNet_model(nn.Module):
    def __init__(self):
        super(WNet_model, self).__init__()
        self.K = 5
        self.output_channels = 3
        self.input_channels = 3
        self.c1 = 64
        self.c2 = 128
        self.c3 = 256
        self.c4 = 512
        self.c5 = 1024

        self.create_enc()
        self.create_dec()

    def forward(self, x):
        # enc names: u_enc1,...7
        # dec names: u_dec1,...7
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        x4 = self.e4(x3)
        x5 = self.e5(x4)
        x6 = self.e6( torch.cat([x4,x5], dim = 1) )
        x7 = self.e7( torch.cat([x3,x6], dim = 1) )
        x8 = self.e8( torch.cat([x2,x7], dim = 1) )
        segmented = self.e9( torch.cat([x1,x8], dim = 1) )

        x10 = self.u1(segmented)
        x11 = self.u2(x10)
        x12 = self.u3(x11)
        x13 = self.u4(x12)
        x14 = self.u5(x13)
        x15 = self.u6( torch.cat([x13,x14], dim = 1) )
        x16 = self.u7( torch.cat([x12,x15], dim = 1) )
        x17 = self.u8( torch.cat([x11,x16], dim = 1) )
        reconstructed = self.u9( torch.cat([x10,x17], dim = 1) )

        return [segmented, reconstructed]

    def create_enc(self):
        self.e1 = nn.Sequential()
        self.e1.add_module('Conv1_1', nn.Conv2d(self.input_channels, self.c1, kernel_size=3, padding=(1, 1)))
        self.e1.add_module('Relu1_2', nn.ReLU(True))
        self.e1.add_module('Batch', nn.BatchNorm2d(self.c1))
        self.e1.add_module('Conv1_3', nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=(1, 1)))
        self.e1.add_module('Relu1_4', nn.ReLU(True))
        self.e1.add_module('Batch', nn.BatchNorm2d(self.c1))

        self.e2 = nn.Sequential()
        self.e2.add_module('Max2_1', nn.MaxPool2d(2, stride=2))
        self.e2.add_module('Conv2_2', nn.Conv2d(self.c1, self.c2, kernel_size=3, padding=(1, 1)))
        self.e2.add_module('Relu2_3', nn.ReLU(True))
        self.e2.add_module('Batch', nn.BatchNorm2d(self.c2))
        self.e2.add_module('Conv2_4', nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=(1, 1)))
        self.e2.add_module('Relu2_5', nn.ReLU(True))
        self.e2.add_module('Batch', nn.BatchNorm2d(self.c2))

        self.e3 = nn.Sequential()
        self.e3.add_module('Max3_1', nn.MaxPool2d(2, stride=2))
        self.e3.add_module('Conv3_2', nn.Conv2d(self.c2, self.c3, kernel_size=3, padding=(1, 1)))
        self.e3.add_module('Relu3_3', nn.ReLU(True))
        self.e3.add_module('Batch', nn.BatchNorm2d(self.c3))
        self.e3.add_module('Conv3_4', nn.Conv2d(self.c3, self.c3, kernel_size=3, padding=(1, 1)))
        self.e3.add_module('Relu3_5', nn.ReLU(True))
        self.e3.add_module('Batch', nn.BatchNorm2d(self.c3))

        self.e4 = nn.Sequential()
        self.e4.add_module('Max4_1', nn.MaxPool2d(2, stride=2))
        self.e4.add_module('Conv4_2', nn.Conv2d(self.c3, self.c4, kernel_size=3, padding=(1, 1)))
        self.e4.add_module('Relu4_3', nn.ReLU(True))
        self.e4.add_module('Batch', nn.BatchNorm2d(self.c4))
        self.e4.add_module('Conv4_4', nn.Conv2d(self.c4, self.c4, kernel_size=3, padding=(1, 1)))
        self.e4.add_module('Relu4_5', nn.ReLU(True))
        self.e4.add_module('Batch', nn.BatchNorm2d(self.c4))

        self.e5 = nn.Sequential()
        self.e5.add_module('Max4_1', nn.MaxPool2d(2, stride=2))
        self.e5.add_module('Conv4_2', nn.Conv2d(self.c4, self.c5, kernel_size=3, padding=(1, 1)))
        self.e5.add_module('Relu4_3', nn.ReLU(True))
        self.e5.add_module('Batch', nn.BatchNorm2d(self.c5))
        self.e5.add_module('Conv4_4', nn.Conv2d(self.c5, self.c5, kernel_size=3, padding=(1, 1)))
        self.e5.add_module('Relu4_5', nn.ReLU(True))
        self.e5.add_module('Batch', nn.BatchNorm2d(self.c5))
        self.e5.add_module('CT4_6', nn.ConvTranspose2d(self.c5, self.c4, kernel_size=2, stride=2))


        self.e6 = nn.Sequential()
        self.e6.add_module('Conv5_1', nn.Conv2d(self.c5, self.c4, kernel_size=3, padding=(1, 1)))
        self.e6.add_module('Relu5_2', nn.ReLU(True))
        self.e6.add_module('Batch', nn.BatchNorm2d(self.c4))
        self.e6.add_module('Conv5_3', nn.Conv2d(self.c4, self.c4, kernel_size=3, padding=(1, 1)))
        self.e6.add_module('Relu5_4', nn.ReLU(True))
        self.e6.add_module('Batch', nn.BatchNorm2d(self.c4))
        self.e6.add_module('CT5_5', nn.ConvTranspose2d(self.c4, self.c3, kernel_size=2, stride=2))

        self.e7 = nn.Sequential()
        self.e7.add_module('Conv6_1', nn.Conv2d(self.c4, self.c3, kernel_size=3, padding=(1, 1)))
        self.e7.add_module('Relu6_2', nn.ReLU(True))
        self.e7.add_module('Batch', nn.BatchNorm2d(self.c3))
        self.e7.add_module('Conv6_3', nn.Conv2d(self.c3, self.c3, kernel_size=3, padding=(1, 1)))
        self.e7.add_module('Relu6_4', nn.ReLU(True))
        self.e7.add_module('Batch', nn.BatchNorm2d(self.c3))
        self.e7.add_module('CT6_5', nn.ConvTranspose2d(self.c3, self.c2, kernel_size=2, stride=2))

        self.e8 = nn.Sequential()
        self.e8.add_module('Conv7_1', nn.Conv2d(self.c3, self.c2, kernel_size=3, padding=(1, 1)))
        self.e8.add_module('Relu7_2', nn.ReLU(True))
        self.e8.add_module('Batch', nn.BatchNorm2d(self.c2))
        self.e8.add_module('Conv7_3', nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=(1, 1)))
        self.e8.add_module('Relu7_4', nn.ReLU(True))
        self.e8.add_module('Batch', nn.BatchNorm2d(self.c2))
        self.e8.add_module('CT7_5', nn.ConvTranspose2d(self.c2, self.c1, kernel_size=2, stride=2))

        self.e9 = nn.Sequential()
        self.e9.add_module('Conv7_1', nn.Conv2d(self.c2, self.c1, kernel_size=3, padding=(1, 1)))
        self.e9.add_module('Relu7_2', nn.ReLU(True))
        self.e9.add_module('Batch', nn.BatchNorm2d(self.c1))
        self.e9.add_module('Conv7_3', nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=(1, 1)))
        self.e9.add_module('Relu7_4', nn.ReLU(True))
        self.e9.add_module('Batch', nn.BatchNorm2d(self.c1))
        self.e9.add_module("Conv7_5", nn.Conv2d(self.c1, self.K, kernel_size=1))
        self.e9.add_module('Soft7_6', nn.Softmax())

    def create_dec(self):
        self.u1 = nn.Sequential()
        self.u1.add_module('Conv1_1', nn.Conv2d(self.K, self.c1, kernel_size=3, padding=(1, 1)))
        self.u1.add_module('Relu1_2', nn.ReLU(True))
        self.u1.add_module('Batch', nn.BatchNorm2d(self.c1))
        self.u1.add_module('Conv1_3', nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=(1, 1)))
        self.u1.add_module('Relu1_4', nn.ReLU(True))
        self.u1.add_module('Batch', nn.BatchNorm2d(self.c1))


        self.u2 = nn.Sequential()
        self.u2.add_module('Max2_1', nn.MaxPool2d(2, stride=2))
        self.u2.add_module('Conv2_2', nn.Conv2d(self.c1, self.c2, kernel_size=3, padding=(1, 1)))
        self.u2.add_module('Relu2_3', nn.ReLU(True))
        self.u2.add_module('Batch', nn.BatchNorm2d(self.c2))
        self.u2.add_module('Conv2_4', nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=(1, 1)))
        self.u2.add_module('Relu2_5', nn.ReLU(True))
        self.u2.add_module('Batch', nn.BatchNorm2d(self.c2))


        self.u3 = nn.Sequential()
        self.u3.add_module('Max3_1', nn.MaxPool2d(2, stride=2))
        self.u3.add_module('Conv3_2', nn.Conv2d(self.c2, self.c3, kernel_size=3, padding=(1, 1)))
        self.u3.add_module('Relu3_3', nn.ReLU(True))
        self.u3.add_module('Batch', nn.BatchNorm2d(self.c3))
        self.u3.add_module('Conv3_4', nn.Conv2d(self.c3, self.c3, kernel_size=3, padding=(1, 1)))
        self.u3.add_module('Relu3_5', nn.ReLU(True))
        self.u3.add_module('Batch', nn.BatchNorm2d(self.c3))


        self.u4 = nn.Sequential()
        self.u4.add_module('Max4_1', nn.MaxPool2d(2, stride=2))
        self.u4.add_module('Conv4_2', nn.Conv2d(self.c3, self.c4, kernel_size=3, padding=(1, 1)))
        self.u4.add_module('Relu4_3', nn.ReLU(True))
        self.u4.add_module('Batch', nn.BatchNorm2d(self.c4))
        self.u4.add_module('Conv4_4', nn.Conv2d(self.c4, self.c4, kernel_size=3, padding=(1, 1)))
        self.u4.add_module('Relu4_5', nn.ReLU(True))
        self.u4.add_module('Batch', nn.BatchNorm2d(self.c4))


        self.u5 = nn.Sequential()
        self.u5.add_module('Max4_1', nn.MaxPool2d(2, stride=2))
        self.u5.add_module('Conv4_2', nn.Conv2d(self.c4, self.c5, kernel_size=3, padding=(1, 1)))
        self.u5.add_module('Relu4_3', nn.ReLU(True))
        self.u5.add_module('Batch', nn.BatchNorm2d(self.c5))
        self.u5.add_module('Conv4_4', nn.Conv2d(self.c5, self.c5, kernel_size=3, padding=(1, 1)))
        self.u5.add_module('Relu4_5', nn.ReLU(True))
        self.u5.add_module('Batch', nn.BatchNorm2d(self.c5))
        self.u5.add_module('CT4_6', nn.ConvTranspose2d(self.c5, self.c4, kernel_size=2, stride=2))


        self.u6 = nn.Sequential()
        self.u6.add_module('Conv5_1', nn.Conv2d(self.c5, self.c4, kernel_size=3, padding=(1, 1)))
        self.u6.add_module('Relu5_2', nn.ReLU(True))
        self.u6.add_module('Batch', nn.BatchNorm2d(self.c4))
        self.u6.add_module('Conv5_3', nn.Conv2d(self.c4, self.c4, kernel_size=3, padding=(1, 1)))
        self.u6.add_module('Relu5_4', nn.ReLU(True))
        self.u6.add_module('Batch', nn.BatchNorm2d(self.c4))
        self.u6.add_module('CT5_5', nn.ConvTranspose2d(self.c4, self.c3, kernel_size=2, stride=2))


        self.u7 = nn.Sequential()
        self.u7.add_module('Conv6_1', nn.Conv2d(self.c4, self.c3, kernel_size=3, padding=(1, 1)))
        self.u7.add_module('Relu6_2', nn.ReLU(True))
        self.u7.add_module('Batch', nn.BatchNorm2d(self.c3))

        self.u7.add_module('Conv6_3', nn.Conv2d(self.c3, self.c3, kernel_size=3, padding=(1, 1)))
        self.u7.add_module('Relu6_4', nn.ReLU(True))
        self.u7.add_module('Batch', nn.BatchNorm2d(self.c3))
        self.u7.add_module('CT6_5', nn.ConvTranspose2d(self.c3, self.c2, kernel_size=2, stride=2))

        self.u8 = nn.Sequential()
        self.u8.add_module('Conv7_1', nn.Conv2d(self.c3, self.c2, kernel_size=3, padding=(1, 1)))
        self.u8.add_module('Relu7_2', nn.ReLU(True))
        self.u8.add_module('Batch', nn.BatchNorm2d(self.c2))

        self.u8.add_module('Conv7_3', nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=(1, 1)))
        self.u8.add_module('Relu7_4', nn.ReLU(True))
        self.u8.add_module('Batch', nn.BatchNorm2d(self.c2))
        self.u8.add_module('CT7_5', nn.ConvTranspose2d(self.c2, self.c1, kernel_size=2, stride=2))

        self.u9 = nn.Sequential()
        self.u9.add_module('Conv7_1', nn.Conv2d(self.c2, self.c1, kernel_size=3, padding=(1, 1)))
        self.u9.add_module('Relu7_2', nn.ReLU(True))
        self.u9.add_module('Batch', nn.BatchNorm2d(self.c1))

        self.u9.add_module('Conv7_3', nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=(1, 1)))
        self.u9.add_module('Relu7_4', nn.ReLU(True))
        self.u9.add_module('Batch', nn.BatchNorm2d(self.c1))
        self.u9.add_module("Conv7_5", nn.Conv2d(self.c1, self.output_channels, kernel_size=1))

