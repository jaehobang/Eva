from eva_storage.baselines.indexing.external.ssd.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from eva_storage.baselines.indexing.external.ssd.vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from eva_storage.baselines.indexing.external.ssd.vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from eva_storage.baselines.indexing.external.ssd.vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from eva_storage.baselines.indexing.external.ssd.vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from eva_storage.baselines.indexing.external.ssd.vision.utils.misc import Timer
from logger import Logger

import cv2
import sys




class SSD_predictor_wrapper:

    def __init__(self, model_path=None, class_names=None):
        self.logger = Logger()
        if model_path is None:
            self.logger.error(f"{__file__}, Model Path is not provided!")
        if class_names is None:
            self.logger.error(f"{__file__}, Class Names are not provided!")

        ## class_names should be in the same order as the trained SSD -- must include BACKGROUND as the 0th class_name
        self.net = create_vgg_ssd(len(class_names), is_test=True)
        self.net.load(model_path)
        self.predictor = create_vgg_ssd_predictor(self.net, candidate_size = 200)
        self.class_names = class_names

    def _convert_labels(self, labels):
        """
        This function aims to convert the labels outputted from the predictor to match the class_names in categories
        :param labels: [0,1,2,3,4...]
        :return: ['BACKGROUND', 'car', 'van', 'bus', 'others'....]
        """
        converted_labels = []
        for label in labels:
            converted_labels.append(self.class_names[label])
        assert(len(converted_labels) == len(labels))
        return converted_labels



    def predict(self, images, categories=None):
        if categories is None:
            self.logger.error(f"{__file__}, must provide categories as list! ex: ['car', 'van']")

        label_dict = {}
        for category in categories:
            label_dict[category] = []
        for image in images:
            boxes, labels, probs = self.predictor.predict(image, 10, 0.4)
            labels_converted = labels

            for category in categories:
                if category in labels_converted:
                    label_dict[category].append(1)
                else:
                    label_dict[category].append(0)

        return label_dict




