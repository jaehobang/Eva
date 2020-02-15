"""
This file defines the process to train overnight and save the model...
@Jaeho Bang
"""




import os
import sys
sys.path.append('../../')

import eva_storage.evaluation.evaluate_ssd as evaluate_ssd
from loaders.uadetrac_loader import UADetracLoader

import os
import logging
import itertools
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR


from eva_storage.external.ssd.vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from eva_storage.external.ssd.vision.ssd.ssd import MatchPriorModified
from eva_storage.external.ssd.vision.ssd.vgg_ssd import create_vgg_ssd
from eva_storage.external.ssd.vision.nn.multibox_loss import MultiboxLoss
from eva_storage.external.ssd.vision.ssd.config import vgg_ssd_config
from eva_storage.external.ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform



DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")




def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, gt_locations, labels = data
        images = images.to(device)
        gt_locations = gt_locations.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        ## need to make sure criterion takes in locations
        regression_loss, classification_loss = criterion(confidence, locations, labels, gt_locations)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num




if __name__ == "__main__":
    # load the data


    loader = UADetracLoader()
    images = loader.load_cached_images()
    labels = loader.load_cached_labels()
    boxes = loader.load_boxes()
    video_start_indices = loader.get_video_start_indices()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        logging.info("Use Cuda.")

    base_net = os.path.join("/nethome/jbang36/eva/eva_storage/external/ssd", "models/vgg16_reducedfc.pth")
    batch_size = 24
    num_workers = 4
    num_epochs = 5000
    checkpoint_folder = 'models/'
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    validation_epochs = 100
    debug_steps = 100

    timer = Timer()

    create_net = create_vgg_ssd
    config = vgg_ssd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean,
                                        config.image_std)
    target_transform = MatchPriorModified(config.priors, config.center_variance,
                                          config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean,
                                   config.image_std)

    ## Load the model

    num_classes = 5  # should this be 4 or 5?
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    base_net_lr = lr
    extra_layers_lr = lr

    print("Base net is frozen..")
    freeze_net_layers(net.base_net)
    params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                             net.regression_headers.parameters(), net.classification_headers.parameters())
    params = [
        {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
        ), 'lr':   extra_layers_lr},
        {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
        )}
    ]

    net.init_from_base_net(base_net)
    ## loading from pretrained model!!
    # pretrained_ssd_dir = '/nethome/jbang36/eva/eva_storage/external/ssd/models/vgg16-ssd-Epoch-149-Loss-3.3744568502269505.pth'
    # net.init_from_pretrained_ssd(pretrained_ssd_dir)

    net.to(DEVICE)

    print("Done loading to GPU")

    ## Load the rest (optimizer, loss function, etc)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=5e-4)

    milestones = [int(v.strip()) for v in "80,100".split(",")]
    scheduler = MultiStepLR(optimizer, milestones=milestones,
                            gamma=0.1, last_epoch=last_epoch)

    ## I don't think you need to normalize.... The network deals with transformations by itself??

    division_point = int(images.shape[0] * 0.8)
    X_train = images[:division_point]
    X_test = images[division_point:]
    y_train = labels['vehicle'][:division_point]
    y_test = labels['vehicle'][division_point:]
    y_train_boxes = boxes[:division_point]
    y_test_boxes = boxes[division_point:]

    ## We need to make val dataset...

    val_division = int(0.8 * len(X_train))
    X_val = X_train[val_division:]
    X_train = X_train[:val_division]
    y_val_boxes = y_train_boxes[val_division:]
    y_train_boxes = y_train_boxes[:val_division]
    y_val = y_train[val_division:]
    y_train = y_train[:val_division]

    train_dataset = evaluate_ssd.UADataset_lite(transform=train_transform, target_transform=target_transform)

    train_dataset.set_x(X_train)
    train_dataset.set_y(y_train)
    train_dataset.set_y_boxes(y_train_boxes)

    val_dataset = evaluate_ssd.UADataset_lite(transform=train_transform, target_transform=target_transform)
    val_dataset.set_x(X_val)
    val_dataset.set_y(y_val)
    val_dataset.set_y_boxes(y_val_boxes)

    test_dataset = evaluate_ssd.UADataset_lite(transform=train_transform, target_transform=target_transform)
    test_dataset.set_x(X_test)
    test_dataset.set_y(y_test)
    test_dataset.set_y_boxes(y_test_boxes)

    ## convert to loader
    batch_size = 24
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=4,
                              shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=4,
                            shuffle=True)

    ## train the model
    ## let's just import utils and do all evaluations here?
    import time

    st = time.time()

    for epoch in range(last_epoch + 1, num_epochs):
        scheduler.step()
        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=debug_steps, epoch=epoch)

        if epoch % validation_epochs == 0 or epoch == num_epochs - 1:
            net.eval()
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0
            num = 0
            for _, data in enumerate(val_loader):
                images, boxes, labels = data
                images = images.to(DEVICE)
                boxes = boxes.to(DEVICE)
                labels = labels.to(DEVICE)
                num += 1
                with torch.no_grad():
                    confidence, locations = net(images)
                    regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
                    loss = regression_loss + classification_loss
                running_loss += loss.item()
                running_regression_loss += regression_loss.item()
                running_classification_loss += classification_loss.item()
            val_loss = running_loss / num
            val_regression_loss = running_regression_loss / num
            val_classification_loss = running_regression_loss / num
            print("epoch", epoch)
            print("  Validation Loss: {v:.4f}".format(v=val_loss))
            print("  Validation Regression Loss: {v:.4f}".format(v=val_regression_loss))
            print("  Validation Classification Loss: {v:.4f}".format(v=val_classification_loss))

            checkpoint_folder = '/nethome/jbang36/eva/eva_storage/external/ssd/models/overnight'
            checkpoint_file_name = "vgg16-ssd:epoch" + str(epoch)
            model_path = os.path.join(checkpoint_folder, checkpoint_file_name)
            net.save(model_path)

    print("Total time to train...", time.time() - st, "seconds")