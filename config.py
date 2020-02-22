import os
import torch

eva_dir = os.path.dirname(os.path.abspath(__file__))
device_number = 0 # main gpu

if torch.cuda.device_count() > 1:
    train_device = torch.device('cuda:0')
    eval_device = torch.device('cuda:1')
elif torch.cuda.device_count() == 1:
    train_device = torch.device('cuda:0')
    eval_device = train_device
else:
    train_device = torch.device('cpu')
    eval_device = torch.device('cpu')

#device = torch.cuda.device(device_number) if torch.cuda.is_available() else torch.device('cpu')


