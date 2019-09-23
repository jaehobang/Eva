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

        self.create_enc()
        self.create_dec()

    def forward(self, x):
        # enc names: u_enc1,...7
        # dec names: u_dec1,...7
        x1 = self.u_enc1(x)
        x2 = self.u_enc2(x1)
        x3 = self.u_enc3(x2)
        x4 = self.u_enc4(x3)
        x34 = torch.cat([x3, x4], dim=1)
        x5 = self.u_enc5(x34)
        x25 = torch.cat([x2, x5], dim=1)
        x6 = self.u_enc6(x25)
        x16 = torch.cat([x1, x6], dim=1)
        x7 = self.u_enc7(x16)

        x8 = self.u_dec8(x7)
        x9 = self.u_dec9(x8)
        x10 = self.u_deca(x9)
        x11 = self.u_decb(x10)
        x1011 = torch.cat([x10, x11], dim=1)
        x12 = self.u_decc(x1011)
        x912 = torch.cat([x9, x12], dim=1)
        x13 = self.u_decd(x912)
        x813 = torch.cat([x8, x13], dim=1)
        x14 = self.u_dece(x813)

        return [x7, x14]

    def create_enc(self):
        self.u_enc1 = nn.Sequential()
        self.u_enc1.add_module('Conv1_1', nn.Conv2d(self.input_channels, 16, kernel_size=3, padding=(1, 1)))
        self.u_enc1.add_module('Relu1_2', nn.ReLU(True))
        self.u_enc1.add_module('Conv1_3', nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1)))
        self.u_enc1.add_module('Relu1_4', nn.ReLU(True))

        self.u_enc2 = nn.Sequential()
        self.u_enc2.add_module('Max2_1', nn.MaxPool2d(2, stride=2))
        self.u_enc2.add_module('Conv2_2', nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1)))
        self.u_enc2.add_module('Relu2_3', nn.ReLU(True))
        self.u_enc2.add_module('Conv2_4', nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1)))
        self.u_enc2.add_module('Relu2_5', nn.ReLU(True))

        self.u_enc3 = nn.Sequential()
        self.u_enc3.add_module('Max3_1', nn.MaxPool2d(2, stride=2))
        self.u_enc3.add_module('Conv3_2', nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1)))
        self.u_enc3.add_module('Relu3_3', nn.ReLU(True))
        self.u_enc3.add_module('Conv3_4', nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1)))
        self.u_enc3.add_module('Relu3_5', nn.ReLU(True))

        self.u_enc4 = nn.Sequential()
        self.u_enc4.add_module('Max4_1', nn.MaxPool2d(2, stride=2))
        self.u_enc4.add_module('Conv4_2', nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1)))
        self.u_enc4.add_module('Relu4_3', nn.ReLU(True))
        self.u_enc4.add_module('Conv4_4', nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1)))
        self.u_enc4.add_module('Relu4_5', nn.ReLU(True))
        self.u_enc4.add_module('CT4_6', nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))

        self.u_enc5 = nn.Sequential()
        self.u_enc5.add_module('Conv5_1', nn.Conv2d(128, 64, kernel_size=3, padding=(1, 1)))
        self.u_enc5.add_module('Relu5_2', nn.ReLU(True))
        self.u_enc5.add_module('Conv5_3', nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1)))
        self.u_enc5.add_module('Relu5_4', nn.ReLU(True))
        self.u_enc5.add_module('CT5_5', nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2))

        self.u_enc6 = nn.Sequential()
        self.u_enc6.add_module('Conv6_1', nn.Conv2d(64, 32, kernel_size=3, padding=(1, 1)))
        self.u_enc6.add_module('Relu6_2', nn.ReLU(True))
        self.u_enc6.add_module('Conv6_3', nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1)))
        self.u_enc6.add_module('Relu6_4', nn.ReLU(True))
        self.u_enc6.add_module('CT6_5', nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2))

        self.u_enc7 = nn.Sequential()
        self.u_enc7.add_module('Conv7_1', nn.Conv2d(32, 16, kernel_size=3, padding=(1, 1)))
        self.u_enc7.add_module('Relu7_2', nn.ReLU(True))
        self.u_enc7.add_module('Conv7_3', nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1)))
        self.u_enc7.add_module('Relu7_4', nn.ReLU(True))
        self.u_enc7.add_module("Conv7_5", nn.Conv2d(16, self.K, kernel_size=1))
        self.u_enc7.add_module('Soft7_6', nn.Softmax())

    def create_dec(self):
        self.u_dec8 = nn.Sequential()
        self.u_dec8.add_module('Conv8_1', nn.Conv2d(self.K, 16, kernel_size=3, padding=(1, 1)))
        self.u_dec8.add_module('Relu8_2', nn.ReLU(True))
        self.u_dec8.add_module('Conv8_3', nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1)))
        self.u_dec8.add_module('Relu8_4', nn.ReLU(True))

        self.u_dec9 = nn.Sequential()
        self.u_dec9.add_module('Max9_1', nn.MaxPool2d(2, stride=2))
        self.u_dec9.add_module('Conv9_2', nn.Conv2d(16, 32, kernel_size=3, padding=(1, 1)))
        self.u_dec9.add_module('Relu9_3', nn.ReLU(True))
        self.u_dec9.add_module('Conv9_4', nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1)))
        self.u_dec9.add_module('Relu9_5', nn.ReLU(True))

        self.u_deca = nn.Sequential()
        self.u_deca.add_module('Maxa_1', nn.MaxPool2d(2, stride=2))
        self.u_deca.add_module('Conva_2', nn.Conv2d(32, 64, kernel_size=3, padding=(1, 1)))
        self.u_deca.add_module('Relua_3', nn.ReLU(True))
        self.u_deca.add_module('Conva_4', nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1)))
        self.u_deca.add_module('Relua_5', nn.ReLU(True))

        self.u_decb = nn.Sequential()
        self.u_decb.add_module('Maxb_1', nn.MaxPool2d(2, stride=2))
        self.u_decb.add_module('Convb_2', nn.Conv2d(64, 128, kernel_size=3, padding=(1, 1)))
        self.u_decb.add_module('Relub_3', nn.ReLU(True))
        self.u_decb.add_module('Convb_4', nn.Conv2d(128, 128, kernel_size=3, padding=(1, 1)))
        self.u_decb.add_module('Relub_5', nn.ReLU(True))
        self.u_decb.add_module('CTb_6', nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))

        self.u_decc = nn.Sequential()
        self.u_decc.add_module('Convc_1', nn.Conv2d(128, 64, kernel_size=3, padding=(1, 1)))
        self.u_decc.add_module('Reluc_2', nn.ReLU(True))
        self.u_decc.add_module('Convc_3', nn.Conv2d(64, 64, kernel_size=3, padding=(1, 1)))
        self.u_decc.add_module('Reluc_4', nn.ReLU(True))
        self.u_decc.add_module('CTc_5', nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2))

        self.u_decd = nn.Sequential()
        self.u_decd.add_module('Convd_1', nn.Conv2d(64, 32, kernel_size=3, padding=(1, 1)))
        self.u_decd.add_module('Relud_2', nn.ReLU(True))
        self.u_decd.add_module('Convd_3', nn.Conv2d(32, 32, kernel_size=3, padding=(1, 1)))
        self.u_decd.add_module('Relud_4', nn.ReLU(True))
        self.u_decd.add_module('CTd_5', nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2))

        self.u_dece = nn.Sequential()
        self.u_dece.add_module('Conve_1', nn.Conv2d(32, 16, kernel_size=3, padding=(1, 1)))
        self.u_dece.add_module('Relue_2', nn.ReLU(True))
        self.u_dece.add_module('Conve_3', nn.Conv2d(16, 16, kernel_size=3, padding=(1, 1)))
        self.u_dece.add_module('Relue_4', nn.ReLU(True))
        self.u_dece.add_module("Conve_5", nn.Conv2d(16, self.output_channels, kernel_size=1))
        self.u_dece.add_module('Softe_6', nn.ReLU())
