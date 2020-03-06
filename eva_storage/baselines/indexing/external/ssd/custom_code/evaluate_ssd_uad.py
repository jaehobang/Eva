import sys

home_dir = '/home/jbang36/eva_jaeho'
sys.path.append(home_dir)

## Need to load the modules
from loaders.uadetrac_loader import UADetracLoader
from eva_storage.logger import Logger

import os
import itertools
import time
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from eva_storage.baselines.indexing.external.ssd.vision.utils.misc import freeze_net_layers
from eva_storage.baselines.indexing.external.ssd.vision.ssd.ssd import MatchPrior
from eva_storage.baselines.indexing.external.ssd.vision.nn.multibox_loss import MultiboxLoss
from eva_storage.baselines.indexing.external.ssd.vision.ssd.config import vgg_ssd_config
from eva_storage.baselines.indexing.external.ssd.vision.ssd.data_preprocessing import TrainAugmentation, TestTransform
from eva_storage.baselines.indexing.external.ssd.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor


import eva_storage.baselines.indexing.external.ssd.custom_code.util_ssd_uad as ssd_utils_custom
import eva_storage.baselines.indexing.external.ssd.vision.utils as ssd_utils_original

import config

DEVICE = config.train_device


## Because we need to do debugging mode, we will put everything on the cpu for now
# DEVICE = torch.device('cpu')


def compute_stats(gt_boxes, proposed_boxes, iou=0.5):
    assert (gt_boxes.size() == proposed_boxes.size())
    iou_list = ssd_utils_original.box_utils.iou_of(gt_boxes, proposed_boxes)
    tmp = iou_list > iou

    tp = torch.sum(iou_list > iou)

    return tp.item(), tmp.size()


def evaluate_naive(test_loader, logger):
    """
    This function returns matches but is naive because there can be multiple detections of the same box
    Therefore, it will have an unnecessarily high accuracy

    :param test_loader: DataLoader object that contains test images
    :return: Accuracy
    """
    all_ground_boxes = []
    all_proposed_boxes = []
    all_confidence = []
    all_labels = []
    all_images = []

    for _, data in enumerate(test_loader):
        images, boxes, labels = data
        images = images.to(DEVICE)
        boxes = boxes.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.no_grad():
            confidence, locations = net(images)
            all_proposed_boxes.append(locations)
            all_confidence.append(confidence)
            all_labels.append(labels)
            all_ground_boxes.append(boxes)
            all_images.append(images)

    # each element in this array will be the batch result

    assert (len(all_proposed_boxes) == len(all_confidence))
    assert (len(all_proposed_boxes) == len(all_labels))
    assert (len(all_proposed_boxes) == len(all_ground_boxes))
    assert (len(all_proposed_boxes) == len(all_images))

    tp_all = 0
    boxes_all = 0

    for i, images in enumerate(all_images):
        predicted_locations = all_proposed_boxes[i].cpu()
        predicted_boxes = ssd_utils_original.box_utils.convert_locations_to_boxes(predicted_locations, vgg_ssd_config.priors,
                                                                         vgg_ssd_config.center_variance,
                                                                         vgg_ssd_config.size_variance)
        predicted_boxes = ssd_utils_original.box_utils.center_form_to_corner_form(predicted_boxes)

        gt_locations = all_ground_boxes[i].cpu()
        gt_boxes = ssd_utils_original.box_utils.convert_locations_to_boxes(gt_locations, vgg_ssd_config.priors,
                                                                  vgg_ssd_config.center_variance,
                                                                  vgg_ssd_config.size_variance)
        gt_boxes = ssd_utils_original.box_utils.center_form_to_corner_form(gt_boxes)

        labels = all_labels[i]
        pos_mask = labels > 0
        predicted_boxes_reshaped = predicted_boxes[pos_mask, :]
        gt_boxes_reshaped = gt_boxes[pos_mask, :]
        tp, all_size = compute_stats(gt_boxes_reshaped, predicted_boxes_reshaped)
        tp_all += tp
        boxes_all += all_size[0]


    acc = 1.0 * tp_all / boxes_all
    logger.info(f"Naive custom_code accuracy gives {acc}")
    return acc


def generate_predictions(test_dataset, logger):
    ## looking at the images, I don't think this is a good measure
    ## because it creates multiple boxes for each object...
    ## so we need to figure out a way to eliminate recurring boxes
    # let's try custom_code method that is already implemented from eval_ssd.py

    predictor = create_vgg_ssd_predictor(net, nms_method="hard", device=DEVICE)
    results = []
    st = time.perf_counter()

    for i in range(len(test_dataset)):
        image = test_dataset.get_image(i)
        boxes, labels, probs = predictor.predict(image) ## might have to change this to .predict_modified(image)
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes  # + 1.0 matlab's indexes start from 1
        ], dim=1))

    results = torch.cat(results)

    logger.info(f"Finished evaluating {len(test_dataset)} images in {time.perf_counter() - st} (sec)")

    class_names = ['car', 'bus', 'others', 'van']
    eval_path = '/nethome/jbang36/eva_jaeho/eva_storage/baselines/indexing/external/ssd/custom_code/evaluation'

    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = os.path.join(eval_path,  f"det_test_{class_name}.txt")
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = str(int(sub[i,0]))
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
        logger.info(f"Finished saving {class_name} results to {prediction_path}")




def compute_statistics(test_dataset, logger):
    aps = []
    class_names = ['car', 'bus', 'others', 'van']

    true_case_stat, all_gb_boxes, all_difficult_cases = ssd_utils_custom.group_annotation_by_class(test_dataset)
    iou_threshold = 0.5

    for class_index, class_name in enumerate(class_names):
        use_2007_metric = True

        ap = ssd_utils_custom.compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            iou_threshold,
            use_2007_metric
        )
        aps.append(ap)

    logger.info(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")
    logger.info(f"Used iou threshold of {iou_threshold}")

    return sum(aps) / len(aps)


def compute_statistics_agnostic(test_dataset, logger):
    aps = []
    class_names = ['car', 'bus', 'others', 'van']

    true_case_stat, all_gb_boxes, all_difficult_cases = ssd_utils_custom.group_annotation_by_class(test_dataset)
    iou_threshold = 0.5

    use_2007_metric = True
    ap = ssd_utils_custom.compute_average_precision_class_agnostic(true_case_stat, all_gb_boxes,
                                                                   all_difficult_cases, class_names,
                                                                   iou_threshold, use_2007_metric)



    logger.info(f"\nAverage Precision Class Agnostic Method: {sum(aps)/len(aps)}")
    logger.info(f"Used iou threshold of {iou_threshold}")

    return sum(aps) / len(aps)








if __name__ == "__main__":
    loader = UADetracLoader()
    logger = Logger()
    logger.info("Starting evaluation........")

    home_dir = '/home/jbang36/eva_jaeho'

    images_test = loader.load_images(dir = os.path.join(home_dir, 'data', 'ua_detrac', '5_images'))
    labels_test, boxes_test = loader.load_labels(dir = os.path.join(home_dir, 'data', 'ua_detrac', '5_xml'))
    labels_test = labels_test['vehicle']

    ### we need to take out frames that don't have labels / boxes
    images_test, labels_test, boxes_test = ssd_utils_custom.filter_input(images_test, labels_test, boxes_test)

    logger.info("Finished loading the UADetrac Dataset")



    ## initialize the training variables
    base_net = os.path.join(home_dir, 'eva_storage/baselines/indexing/external/ssd/', "models/vgg16_reducedfc.pth")
    batch_size = 24
    num_workers = 4
    num_epochs = 200
    lr = 1e-3
    momentum = 0.9
    weight_decay = 5e-4
    validation_epochs = 5
    debug_steps = 100

    ## create the net and transforms
    create_net = create_vgg_ssd
    config = vgg_ssd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean,
                                        config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean,
                                   config.image_std)

    ## Load the model
    num_classes = 5  ## is this right?? there is the background.....
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
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    pretrained_ssd_directory = '/nethome/jbang36/eva_jaeho/eva_storage/baselines/indexing/external/ssd/models/vgg16-ssd:epoch199'

    net.init_from_pretrained_ssd(pretrained_ssd_directory)

    net.to(DEVICE)
    logger.info("Finished loading the model and dataset")

    ## Load the rest (optimizer, loss function, etc)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=5e-4)

    milestones = [int(v.strip()) for v in "80,100".split(",")]
    scheduler = MultiStepLR(optimizer, milestones=milestones,
                            gamma=0.1, last_epoch=last_epoch)

    logger.info("Finished creating external modules")

    dataset_test = ssd_utils_custom.UADataset_lite(transform=test_transform, target_transform=target_transform)
    dataset_test.set_images(images_test)
    dataset_test.set_labels(labels_test)
    dataset_test.set_boxes(boxes_test)

    #test_loader = DataLoader(dataset_test, batch_size, num_workers=4, shuffle=True)

    logger.info("Finished creating data loader for model")

    ## do code custom_code...

    #evaluate_naive(test_loader, logger)
    generate_predictions(dataset_test, logger)
    compute_statistics(dataset_test, logger)
















