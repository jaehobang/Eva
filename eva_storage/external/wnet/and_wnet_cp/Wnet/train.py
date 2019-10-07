import torch
import numpy as np
from eva_storage.external.wnet.and_wnet_cp.Wnet.configure import Config
from eva_storage.external.wnet.and_wnet_cp.Wnet.model import WNet
#from eva_storage.external.wnet.and_wnet_cp.Wnet.DataLoader import DataLoader
from eva_storage.external.wnet.and_wnet_cp.Wnet.Ncuts import NCutsLoss
from loaders.uadetrac_loader import LoaderUADetrac
import torch.utils.data

import time
import os

config = Config()
if __name__ == '__main__':
    loader = LoaderUADetrac()
    images = loader.load_cached_images()

    X = images.astype(np.float)
    train_data = torch.from_numpy(X).float()

    N, H, W, C = images.shape
    train_data = train_data.permute(0, 3, 1, 2)
    batch_size = 64
    assert (train_data.size(1) == 3)
    dataloader = torch.utils.data.DataLoader(train_data,
                                               shuffle=True, batch_size=batch_size,
                                               num_workers=1, drop_last=True)


    #model = torch.nn.DataParallel(Net(True))
    model = torch.nn.DataParallel(WNet())
    model.cuda()
    #model.to(device)
    model.train()
    #optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr = config.init_lr)
    #reconstr = torch.nn.MSELoss().cuda(config.cuda_dev)
    Ncuts = NCutsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iter, gamma=config.lr_decay)
    for epoch in range(config.max_iter):
        print("Epoch: "+str(epoch+1))
        scheduler.step()
        Ave_Ncuts = 0.0
        #Ave_Rec = 0.0
        t_load = 0.0
        t_forward = 0.0
        t_loss = 0.0
        t_backward = 0.0
        t_adjust = 0.0
        t_reset = 0.0
        t_inloss = 0.0
        for step,[x,w] in enumerate(dataloader):
            #NCuts Loss
            #tick = time.time()
            x = x.cuda()
            w = w.cuda()
            #for m in torch.arange(50,70,dtype=torch.long):

            #    print(m)
            #    for n in torch.arange(50,70,dtype= torch.long):
            #        print(w[5,0,m,n])
            sw = w.sum(-1).sum(-1)
            #t_load += time.time()-tick
            #tick = time.time()
            optimizer.zero_grad()
            pred,pad_pred = model(x)
            #t_forward += time.time()-tick
            #pred.cuda()
            #tick = time.time()
            ncuts_loss = Ncuts(pred,pad_pred,w,sw)
            ncuts_loss = ncuts_loss.sum()/config.BatchSize 
            #t_loss += time.time()-tick
            #tick = time.time()
            Ave_Ncuts = (Ave_Ncuts * step + ncuts_loss.item())/(step+1)
            #t_reset += time.time()-tick
            #tick = time.time()
            ncuts_loss.backward()
            #t_backward += time.time()-tick
            #tick = time.time()
            optimizer.step()
            #t_adjust += time.time()-tick
            #Reconstruction Loss
            '''pred,rec_image = model(x)
            rec_loss = reconstr(rec_image,x)
            Ave_Rec = (Ave_Rec * step + rec_loss.item())/(step+1)
            optimizer.zero_grad()
            rec_loss.backward()
            optimizer.step()'''
        #t_total = t_load+t_reset+t_forward+t_loss+t_backward+t_adjust
        print("Ncuts loss: "+str(Ave_Ncuts))#+";total time: "+str(t_total)+";forward: "+str(t_forward/t_total)+";loss: "+str(t_loss/t_total)+";backward: "+str(t_backward/t_total)+";adjust: "+str(t_adjust/t_total)+";reset&load: "+str(t_reset/t_total)+"&"+str(t_load/t_total)+"loss: "+str(t_loss)+" / "+str(t_inloss))
        #print("Reconstruction loss: "+str(Ave_Rec))
        if (epoch+1)%500 == 0:
            localtime = time.localtime(time.time())
            checkname = './checkpoints'
            if not os.path.isdir(checkname):
                os.mkdir(checkname)
            checkname+='/checkpoint'
            for i in range(1,5):
                checkname+='_'+str(localtime[i])
            checkname += '_epoch_'+str(epoch+1)
            with open(checkname,'wb') as f:
                torch.save({
                    'epoch': epoch +1,
                    'state_dict': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'Ncuts': Ave_Ncuts#,
                    #'recon': Ave_Rec
                    },f)
            print(checkname+' saved')
