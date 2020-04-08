"""UAD Dataset Classes

Original author: Jaeho Bang


"""
from .config import HOME
import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from logger import Logger

UAD_CLASSES = (  # always index 0
    'car', 'bus', 'others', 'van')

# note: if you used our download scripts, this should be right
UAD_ROOT = osp.join(HOME, "eva_jaeho", "data", "ua_detrac")

#UAD_CLASSES_W_BACK = ('BACKGROUND', 'car', 'bus', 'others', 'van')




###TODO: Need to modify these functions!!!
class UADAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """

    def __call__(self, boxes, labels, width, height):
        """
        This function will be called per frame
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []


        ## we should also convert the string labels to indices

        for i, box in enumerate(boxes):
            final_box = list(np.array(box) / scale)
            label_num = UAD_CLASSES.index(labels[i])
            assert(label_num >= 0 and label_num < len(UAD_CLASSES))
            final_box.append(label_num)
            res += [final_box]

        return res  # [[xmin, ymin, xmax, ymax, label_idx], ... ]


class UADDetection(data.Dataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        set_name (string): Name of the specific set of COCO images.
        transform (callable, optional): A function/transform that augments the
                                        raw images`
        target_transform (callable, optional): A function/transform that takes
        in the target (bbox) and transforms it.
    """

    def __init__(self, transform=None,
                 target_transform=UADAnnotationTransform(), dataset_name='UA-DETRAC', eval = False):

        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name

        self.root = UAD_ROOT
        self.image_width = -1
        self.image_height = -1
        self.logger = Logger()
        self.eval = eval


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target).
                   target is the object returned by ``coco.loadAnns``.
        """
        im, gt, h, w = self.pull_item(index)
        return im, gt


    def set_images(self, images):
        self.X_train = images
        self.image_width = self.X_train.shape[1]
        self.image_height = self.X_train.shape[2]


    def set_labels(self, labels):
        self.y_train = labels


    def set_boxes(self, boxes):
        ## let's normalize the boxes...so that they are optimal for training
        self.y_train_boxes = boxes

    def get_boxes(self, id):
        return np.array(self.y_train_boxes[id])

    def get_labels(self, id):
        return np.array(self.y_train[id])


    def pull_item(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, height, width).
                   target is the object returned by ``coco.loadAnns``.
        """



        img = self.X_train[index]
        height, width, _ = img.shape
        labels = self.y_train[index]
        boxes = self.y_train_boxes[index]
        if self.target_transform is not None:
            target = self.target_transform(boxes, labels, width, height)
            ## TODO: will there be cases where you don't use the target transform?? The original target seems to be ETree of the annotation
        if self.transform is not None:
            ## we expect this code to run only if target_trans
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4],
                                                target[:, 4])
            # to rgb -- we have to do this bc cv2 loads BRG instead of RGB
            img = img[:, :, (2, 1, 0)]

            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width



    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            cv2 img
        '''
        return self.X_train[index] ## note this images is BRG


    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        ## we need to convert the annotation into this format
        ret = [str(index)]
        targets = []
        for i, label in enumerate(self.y_train[index]):
            targets.append((label, self.y_train_boxes[index][i]))
        ret.append(targets)
        return ret

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


    def __len__(self):
        return len(self.X_train)












