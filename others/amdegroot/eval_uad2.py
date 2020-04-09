"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""
from __future__ import print_function

import sys
home_dir = '/home/jbang36/eva_jaeho'
sys.path.append(home_dir)

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from others.amdegroot.ssd import build_ssd

from others.amdegroot.data.uad import UAD_ROOT, UADAnnotationTransform, UADDetection
from others.amdegroot.data.uad import UAD_CLASSES as labelmap
from others.amdegroot.data import BaseTransform
from loaders.uadetrac_loader import UADetracLoader
from logger import Logger


import os
import time
import argparse
import numpy as np
import pickle



if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_UAD_0408_90000.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--uad_root', default=UAD_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

#annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
#imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
#imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
#                          'Main', '{:s}.txt')
#YEAR = '2007'
#devkit_path = args.voc_root + 'VOC' + YEAR
## I guess we can say the devkit path is where ua_detrac is saved.... so
devkit_path = args.uad_root

dataset_mean = (104, 117, 123)
set_type = 'test'

logger = Logger()

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    """
    Returns the results file template
    :param image_set: name of dataset (str)
    :param cls: the corresponding class of interest
    :return:
    """
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = os.path.join(devkit_path, 'results')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    path = os.path.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    """
    This file writes the detection results to a txt file so that eval code can run against it

    :param all_boxes:
    :param dataset:
    :return:
    """
    for cls_ind, cls in enumerate(labelmap):
        print('Writing {:s} UAD results file'.format(cls))
        filename = get_voc_results_file_template(set_type, cls)
        with open(filename, 'wt') as f:
            ## in the case of uad, image id is same as index...
            for i in range(len(dataset)):
                dets = all_boxes[cls_ind+1][i] ## retrieve the detections of boxes for a given image for a given class
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(str(i), dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1)) ## write the detections



def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:True).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap



def detect_all_boxes(net, dataset, output_dir):
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap)+1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    det_file = os.path.join(output_dir, 'detections.pkl')

    for i in range(num_images):
        im, gt, h, w = dataset.pull_item(i)

        x = Variable(im.unsqueeze(0))
        if args.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        detections = net(x).data
        detect_time = _t['im_detect'].toc(average=False)

        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].cpu().numpy()
            cls_dets = np.hstack((boxes.cpu().numpy(),
                                  scores[:, np.newaxis])).astype(np.float32,
                                                                 copy=False)
            all_boxes[j][i] = cls_dets

        print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                    num_images, detect_time))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    return all_boxes



def group_annotation_by_class(dataset):
    all_class_recs = {} ### YOU DONT DO BACKGROUND -- {'car': {image_id: {'bbox': np.array(), 'difficult': np.array(np.bool), 'det': [False] * length}}
    nposes = {}

    for cls_index, cls in enumerate(labelmap):
        all_class_recs[cls] = {}
        nposes[cls] = 0


    for image_id in range(len(dataset)):
        bboxes = dataset.get_boxes(image_id)
        labels = dataset.get_labels(image_id) ## ['car', 'car', 'car', 'bus', ...]



        for label_id, label in enumerate(labelmap):
            all_class_recs[label][image_id] = {}

            relevant_boxes = (labels == label)

            all_class_recs[label][image_id]['bbox'] = bboxes[relevant_boxes]
            all_class_recs[label][image_id]['difficult'] = np.array([False] * sum(relevant_boxes)) ## the sum signifies how many there are
            all_class_recs[label][image_id]['det'] = [False] * sum(relevant_boxes)

            nposes[label] += sum(relevant_boxes)

    logger.info(all_class_recs)
    logger.info(nposes)
    return all_class_recs, nposes



def do_python_eval(all_class_recs, nposes, output_dir='output', use_07=True):
    """
    This function simply wraps the voc_eval function that does the main evaluation
    This function wraps and simply presents the AP values for each given class
    :param true_case_stats -- I have no idea what this is yet
    :param all_gt_boxes -- ground truth boxes that are organized in terms of image indexes
    :param all_difficult_case -- I have no idea what this is yet
    :param output_dir: directory to where detection files are located
    :param use_07: whether to use 07 evaluation format
    :return:
    """
    #cachedir = os.path.join(devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(labelmap):
        filename = get_voc_results_file_template(set_type, cls)

        rec, prec, ap = voc_eval(filename, cls, all_class_recs[cls], nposes[cls], ovthresh=0.5, use_07_metric=use_07_metric)

        ########## we will make sure everything goes as expected up to here
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)

    ### we should ignore -1 values.....
    aps = np.array(aps)
    relevant_aps_indices = aps != -1
    aps = aps[relevant_aps_indices]

    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_eval(detpath, classname, class_recs, npos, ovthresh=0.5, use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    classname: Category name (duh)
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
    # assumes detections are in detpath.format(classname)


    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0] for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            ## we need to convert the image ids into numbers!!!
            R = class_recs[int(image_ids[d])]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        ### just in case npos is zero

        if npos == 0:
            rec = -1.
            prec = -1.
            ap = -1.
        else:
            rec = tp / float(npos)

            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)


            ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


if __name__ == '__main__':
    # load net
    #args.trained_model = 'weights/sadfasdfasdf'
    """
    This file is the optimized version of eval_uad
    eval_uad now works"""


    num_classes = len(labelmap) + 1                      # +1 for background
    net = build_ssd('test', 300, num_classes)            # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    loader = UADetracLoader()

    #images = loader.load_images(dir = os.path.join(home_dir, 'data', 'ua_detrac', '5_images'))
    #labels, boxes = loader.load_labels(dir = os.path.join(home_dir, 'data', 'ua_detrac', '5_xml'))

    images = loader.load_cached_images(name = 'uad_test_images.npy', vi_name = 'uad_test_vi.npy')
    labels = loader.load_cached_labels(name = 'uad_test_labels.npy')
    boxes = loader.load_cached_boxes(name = 'uad_test_boxes.npy')
    labels = labels['vehicle']

    images, labels, boxes = loader.filter_input3(images, labels, boxes)

    ## let's sample images
    images = images[::10]
    labels = labels[::10]
    boxes = boxes[::10]

    dataset = UADDetection(transform=BaseTransform(300, dataset_mean), target_transform=UADAnnotationTransform())
    dataset.set_images(images)
    dataset.set_labels(labels)
    dataset.set_boxes(boxes)


    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    """
    Steps:
    1. detect all the boxes
    2. Save the results in ssd300_uad/test/detections.pkl
    3. Save the results in .txt
    4. Run the evaluation code by loading the txt files"""

    output_dir = get_output_dir('ssd300_uad', set_type)
    box_list = detect_all_boxes(net, dataset, output_dir)

    ## TODO: we should examine the format of the box_list!!!
    ## I expect it to be a 2d list [class_index][images_index]

    write_voc_results_file(box_list, dataset)  ## we need to do this because this is how the eval code runs evaluations

    ## the files are saved in det_test_car.txt, det_test_bus.txt,....
    ## individual rows are in the format of [image_index, confidence, left, top, right, bottom]


    all_class_recs, nposes = group_annotation_by_class(dataset)
    ## for labels do we need to add one or not

    ## TODO: we need to examine what true_case_stats, all_difficult_case is... and their corresponding formats


    do_python_eval(all_class_recs, nposes, output_dir)

