import os
import torch

eva_dir = os.path.dirname(os.path.abspath(__file__))
device_number = 1 #alt GPU
device = torch.cuda.device(device_number) if torch.cuda.is_available() else torch.device('cpu')
