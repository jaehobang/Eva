import sys
home_dir = '/home/jbang36/eva_jaeho'
sys.path.append(home_dir)


from others.amdegroot.utils.augmentations import SSDAugmentation
from others.amdegroot.layers.modules import MultiBoxLoss
from others.amdegroot.ssd import build_ssd
import os
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import argparse
import math

from logger import Logger
from others.amdegroot.data.__init__ import detection_collate
from others.amdegroot.data.coco import COCO_ROOT, COCODetection
from others.amdegroot.data.voc0712 import VOC_ROOT, VOCDetection
from others.amdegroot.data.uad import UAD_ROOT, UADDetection
from others.amdegroot.data.config import *
from loaders.uadetrac_loader import UADetracLoader

logger = Logger()


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='UAD', choices=['UAD', 'VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=UAD_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

### TODO: Debugging parameters
### we won't be using cuda for debugging purposes!!
args.dataset = 'UAD'
use_gpu = True
args.batch_size = 32
save_dir = 'weights/ssd300_UAD_0408_'
args.lr = 1e-3
#args.cuda = False
#args.batch_size = 1



if use_gpu and torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'COCO':
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco
        dataset = COCODetection(root=args.dataset_root,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif args.dataset == 'VOC':
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root=args.dataset_root,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif args.dataset == 'UAD':
        logger.info("We are loading UADetrac!!")
        cfg = uad
        dataset = UADDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
        loader = UADetracLoader()
        images = loader.load_cached_images(name='uad_train_images.npy', vi_name='uad_train_vi.npy')
        boxes = loader.load_cached_boxes(name = 'uad_train_boxes.npy')
        labels = loader.load_cached_labels(name = 'uad_train_labels.npy')
        labels = labels['vehicle']
        images, labels, boxes = loader.filter_input3(images, labels, boxes)
        dataset.set_images(images)
        dataset.set_labels(labels)
        dataset.set_boxes(boxes)



    if args.visdom:
        import visdom
        viz = visdom.Visdom()

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if args.cuda:
        net = net.cuda()

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
    criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                             False, args.cuda)

    net.train()
    # loss counters
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset) // args.batch_size
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title = 'SSD.PyTorch on ' + dataset.name
        vis_legend = ['Loc Loss', 'Conf Loss', 'Total Loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    iteration = 0
    start_epoch = 0
    end_epoch = 120
    #for iteration in range(args.start_iter, cfg['max_iter']):
    for epoch in range(start_epoch, end_epoch):
        for i, data_pack in enumerate(data_loader):
            ### MODIFICATION: we next(batch_iterator) doesn't work because workers end when we finish one iteration
            ###               so we will be making a loop that uses epochs instead of iterations and convert these numbers

            images, targets = data_pack
            iteration += 1
            if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
                update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None,
                                'append', epoch_size)
                # reset epoch loss counters
                loc_loss = 0
                conf_loss = 0
                epoch += 1

            if iteration in cfg['lr_steps']:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            # load train data
            #images, targets = next(batch_iterator)

            if args.cuda:
                images = images.cuda()
                targets = [ann.cuda() for ann in targets]
            else:
                #images = images
                targets = [ann for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets) ##out is loc_data, conf_data, prior
            if math.isnan(loss_l.item()):
                logger.error(f"Curr iter: {iteration}, Localization loss is NAN {loss_l.item()}")
            if math.isnan(loss_c.item()):
                logger.error(f"Curr iter: {iteration}, Localization loss is NAN {loss_c.item()}")


            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            ### MODIFICATION: 3/16
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
            #loc_loss += loss_l.data[0]
            #conf_loss += loss_c.data[0]

            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')

            if args.visdom:
                update_vis_plot(iteration, loss_l.item(), loss_c.item(),
                                iter_plot, epoch_plot, 'append')

            if iteration != 0 and iteration % 10000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), save_dir +
                           repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(),
               args.save_folder + '' + args.dataset + '.pth')


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


def create_vis_plot(_xlabel, _ylabel, _title, _legend):
    return viz.line(
        X=torch.zeros((1,)).cpu(),
        Y=torch.zeros((1, 3)).cpu(),
        opts=dict(
            xlabel=_xlabel,
            ylabel=_ylabel,
            title=_title,
            legend=_legend
        )
    )


def update_vis_plot(iteration, loc, conf, window1, window2, update_type,
                    epoch_size=1):
    viz.line(
        X=torch.ones((1, 3)).cpu() * iteration,
        Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win=window1,
        update=update_type
    )
    # initialize epoch plot on first iteration
    if iteration == 0:
        viz.line(
            X=torch.zeros((1, 3)).cpu(),
            Y=torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win=window2,
            update=True
        )


if __name__ == '__main__':
    train()
