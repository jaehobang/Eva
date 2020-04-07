"""
This file implements the dataset loading methods for UA-detrac
If any problem occurs, please email jaeho.bang@gmail.com


@Jaeho Bang

"""

import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2

from loaders import TaskManager
from loaders.abstract_loader import AbstractLoader
import warnings
import argparse
import time

from logger import Logger, LoggingLevel

parser = argparse.ArgumentParser(description='Define arguments for loader')
parser.add_argument('--image_path', default='small-data', help='Define data folder within eva/data/uadetrac')
parser.add_argument('--anno_path', default='small-annotations', help='Define annotation folder within eva/data/uadetrac')
parser.add_argument('--cache_path', default='npy_files', help='Define save folder for images, annotations, boxes')
parser.add_argument('--cache_image_name', default='ua_detrac_images.npy', help='Define filename for saving and loading cached images')
parser.add_argument('--cache_label_name', default='ua_detrac_labels.npy', help='Define filename for saving and loading cached labels')
parser.add_argument('--cache_box_name', default='ua_detrac_boxes.npy', help='Define filename for saving and loading cached boxes')
parser.add_argument('--cache_vi_name', default='ua_detrac_vi.npy', help='Define filename for saving and loading cached video indices')
args = parser.parse_args()

# Make this return a dictionary of label to data for the whole dataset

class UADetracLoader(AbstractLoader):
    def __init__(self, image_width = 300, image_height = 300):
        self.data_dict = {}
        self.label_dict = {}
        self.vehicle_type_filters = ['car', 'van', 'bus', 'others']
        self.speed_filters = [40, 50, 60, 65, 70]
        self.intersection_filters = ["pt335", "pt342", "pt211", "pt208"]
        self.color_filters = ['white', 'black', 'silver', 'red']

        ## original image height = 540
        ## original image width = 960
        self.image_width = image_width
        self.image_height = image_height
        self.image_channels = 3
        self.task_manager = TaskManager.TaskManager()
        self.images = None
        self.labels = None
        self.boxes = None
        self.eva_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.video_start_indices = np.array([])
        self.logger = Logger()


    def load_video(self, dir:str):
        """
        This function is not needed for ua_detrac
        Should never be called
        :return: None
        """
        return None


    def load_boxes(self, dir:str):
        if self.boxes is None:
            self.logger.info("Running load_labels first to get the boxes...")
            self.load_labels(dir)
        return self.get_boxes()



    def debugMode(self, mode = False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)

    def load_images_debug(self, dir: str = None, image_size=None):
        """
        This function simply loads image of given image
        :return: image_array (numpy)
        """
        if image_size is not None:
            self.image_height = image_size
            self.image_width = image_size

        if dir == None:
            dir = os.path.join(self.eva_dir, 'data', 'ua_detrac', args.image_path)

        file_names = []
        video_start_indices = []

        mvi_directories = os.listdir(dir)
        mvi_directories.sort()

        self.logger.debug(mvi_directories)



        # I also need to

        return



    def load_images(self, dir:str = None, image_size=None):
        """
        This function simply loads image of given image
        :return: image_array (numpy)
        """
        if image_size is not None:
            self.image_height = image_size
            self.image_width = image_size

        if dir == None:
            dir = os.path.join(self.eva_dir, 'data', 'ua_detrac', args.image_path)


        file_names = []
        video_start_indices = [0]
        video_length_indices = []

        mvi_directories = os.listdir(dir)
        mvi_directories.sort()



        for mvi_dir in mvi_directories:
            files = os.listdir(os.path.join(dir, mvi_dir))
            if files == []:
                self.logger.info(f"Directory {os.path.join(dir, mvi_dir)} doesn't contain any files!!")
                continue
            files.sort()
            video_length_indices.append(len(files))
            for file in files:
                file_names.append(os.path.join(dir, mvi_dir, file))

        self.logger.debug(f"Number of directories: {len(mvi_directories)}")
        self.logger.debug(f"Directories are: {mvi_directories}")
        st = time.perf_counter()

        self.images = np.ndarray(shape=(
        len(file_names), self.image_height, self.image_width, self.image_channels),
                               dtype=np.uint8)


        for i in range(len(file_names)):
              file_name = file_names[i]
              img = cv2.imread(file_name)
              img = cv2.resize(img, (self.image_width, self.image_height))
              self.images[i] = img

        for i, length in enumerate(range(len(video_length_indices))):
            video_start_indices.append( video_length_indices[i] + video_start_indices[i] )
        self.video_start_indices = np.array(video_start_indices)

        self.logger.info(f"Total time to load {len(file_names)} images: {time.perf_counter() - st} (sec)")

        return self.images


    def load_labels(self, dir:str = None):
        """
        Loads vehicle type, speed, color, and intersection of ua-detrac
        vehicle type, speed is given by the dataset
        color, intersection is derived from functions built-in
        :return: labels
        """

        if dir == None:
            dir = os.path.join(self.eva_dir, 'data', 'ua_detrac', args.anno_path)
        results = self._load_XML(dir)
        if results is not None:
            vehicle_type_labels, speed_labels, color_labels, intersection_labels = results
            self.labels = {'vehicle': vehicle_type_labels, 'speed': speed_labels,
                    'color': color_labels, 'intersection': intersection_labels}

            return self.labels, self.boxes
        else:
            return None

    ##### Updated 3/3/2020 -- Filters the image input because the loaded data can contain None

    def filter_input3(self, images_train, labels_train, boxes_train):
        length = len(images_train)

        ## first determine count of non None frame
        count = 0
        for i in range(length):
            if labels_train[i] is not None:
                count += 1

        new_images_train = np.ndarray(
            shape=(count, images_train.shape[1], images_train.shape[2], images_train.shape[3]))
        new_labels_train = []
        new_boxes_train = []

        index = 0
        for i, elem in enumerate(labels_train):
            if elem is not None:
                new_images_train[index] = images_train[i]
                index += 1
                new_labels_train.append(elem)
                new_boxes_train.append(boxes_train[i])

        assert (len(new_images_train) == len(new_labels_train))
        assert (len(new_images_train) == len(new_boxes_train))

        return new_images_train, new_labels_train, new_boxes_train


    def filter_input(self, labels_train, boxes_train):

        length = len(labels_train)
        ## first determine count of non None frame
        count = 0
        for i in range(length):
            if labels_train[i] is not None:
                count += 1


        new_labels_train = []
        new_boxes_train = []

        for i, elem in enumerate(labels_train):
            if elem is not None:
                new_labels_train.append(elem)
                new_boxes_train.append(boxes_train[i])

        assert(len(new_labels_train) == len(new_boxes_train))

        return new_labels_train, new_boxes_train



    def get_boxes(self):
        """
        This function must be run after load_labels
        :param dir:
        :return:
        """
        if self.boxes is None:
            self.logger.error("get_labels must be run before this function to obtain the boxes!")
            raise ValueError

        self.logger.debug(f"Total number of frames loaded: {len(self.boxes)}")
        return self.boxes



    def get_video_start_indices(self):
        """
        This function returns the starting indexes for each video bc uadetrac has numerous videos of different perspectives
        :return: python list with starting indexes saved
        """
        return self.video_start_indices


    def save_images(self, name = args.cache_image_name, vi_name = args.cache_vi_name):
        # we need to save the image / video start indexes
        # convert list to np.array
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        save_dir_vi = os.path.join(self.eva_dir, 'data', args.cache_path, vi_name)
        if self.images is None:
            self.logger.error("No image loaded, call load_images() first")
        elif type(self.images) is np.ndarray:
            np.save(save_dir, self.images)
            np.save(save_dir_vi, np.array(self.video_start_indices))
            self.logger.info(f"saved images to {save_dir}")
            self.logger.info(f"saved video indices to {save_dir_vi}")
        else:
            self.logger.error("Image array type is not np.....cannot save")



    def save_labels(self, name=args.cache_label_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        if self.labels is None:
            self.logger.error("No labels loaded, call load_labels() first")

        elif type(self.labels) is dict:
            np.save(save_dir, self.labels, allow_pickle=True)
            self.logger.info(f"saved labels to {save_dir}")

        else:
            self.logger.error(f"Expected labels type to be dict, got {type(self.labels)}")



    def save_boxes(self, name=args.cache_box_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        if self.images is None:
            self.logger.error("No labels loaded, call load_boxes() first")

        elif type(self.images) is np.ndarray:
            np.save(save_dir, self.boxes)
            self.logger.info(f"saved boxes to {save_dir}")

        else:
            self.logger.error("Labels type is not np....cannot save")


    def load_cached_images(self, name=args.cache_image_name, vi_name=args.cache_vi_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        save_dir_vi = os.path.join(self.eva_dir, 'data', args.cache_path, vi_name)
        self.images = np.load(save_dir)
        self.video_start_indices = np.load(save_dir_vi)
        return self.images

    def load_cached_boxes(self, name=args.cache_box_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        self.boxes = np.load(save_dir, allow_pickle = True)
        return self.boxes

    def load_cached_labels(self, name = args.cache_label_name):
        save_dir = os.path.join(self.eva_dir, 'data', args.cache_path, name)
        labels_pickeled = np.load(save_dir, allow_pickle = True)
        self.labels = labels_pickeled.item()
        return self.labels



    def _convert_speed(self, original_speed):
        """
        TODO: Need to actually not use this function, because we need to find out what the original speed values mean
        TODO: However, in the meantime, we will use this extrapolation....
        :param original_speed:
        :return: converted_speed
        """
        speed_range = [0.0, 20.0]
        converted_range = [0.0, 100.0]

        return original_speed * 5






    def _load_XML(self, directory):
        """
        UPDATE: vehicle colors can now be extracted through the xml files!!! We will toss the color generator
        :param directory:
        :return:
        """
        car_labels = []
        speed_labels = []
        color_labels = []
        intersection_labels = []
        self.boxes = None

        width = self.image_width
        height = self.image_height
        original_height = 540
        original_width = 960

        boxes_dataset = []

        if self.images is None:
            warnings.warn("Must load image before loading labels...returning", Warning)
            return None

        self.logger.debug(f"walking {directory},for xml parsing")
        for root, subdirs, files in os.walk(directory):
            if '.ipy' in root:
                continue

            ## need to take out swp files
            for filename in files:
                if ".swp" in filename:
                    files.remove(filename)
                elif ".swo" in filename:
                    files.remove(filename)

            ##as a sanity check, let's print files
            self.logger.debug(f"before sorting operation {files}")

            files.sort()
            self.logger.debug(f"{root} {subdirs} {files}")

            self.logger.debug(f"length of video start indices: {len(self.video_start_indices)}")
            self.logger.debug(self.video_start_indices)
            self.logger.debug(f"length of files {len(files)}")
            assert (len(self.video_start_indices) == len(files) + 1)

            for i, file in enumerate(files):
                file_path = os.path.join(root, file)
                if ".swp" in file_path:
                    continue
                tree = ET.parse(file_path)
                tree_root = tree.getroot()

                car_labels_file = []
                speed_labels_file = []
                color_labels_file = []
                intersection_labels_file = []
                boxes_labels_file = []
                curr_frame_num = 0

                for frame in tree_root.iter('frame'):

                    prev_frame_num = curr_frame_num
                    curr_frame_num = int(frame.attrib['num'])
                    ## updated 1/21/2020 to accomdate xml files that doesn't have annotations in the middle
                    if len(car_labels_file) + 1 != curr_frame_num:
                        boxes_labels_file.extend([None] * (curr_frame_num - prev_frame_num - 1))
                        car_labels_file.extend([None] * (curr_frame_num - prev_frame_num - 1))
                        speed_labels_file.extend([None] * (curr_frame_num - prev_frame_num - 1))
                        color_labels_file.extend([None] * (curr_frame_num - prev_frame_num - 1))
                        intersection_labels_file.extend([None] * (curr_frame_num - prev_frame_num - 1))

                    boxes_per_frame = []
                    for box in frame.iter('box'):
                        left = int(float(box.attrib['left']) * width / original_width)
                        top = int(float(box.attrib['top']) * height / original_height)
                        right = int((float(box.attrib['left']) + float(box.attrib['width'])) * width / original_width)
                        bottom = int((float(box.attrib['top']) + float(box.attrib['height'])) * height / original_height)

                        boxes_per_frame.append((left, top, right, bottom))

                    car_per_frame = []
                    speed_per_frame = []
                    color_per_frame = []
                    intersection_per_frame = []

                    for att in frame.iter('attribute'):
                        if (att.attrib['vehicle_type']):
                            car_per_frame.append(att.attrib['vehicle_type'])
                        if (att.attrib['speed']):
                            speed_per_frame.append(self._convert_speed(float(att.attrib['speed'])))
                        if ('color' in att.attrib.keys()):
                            color_per_frame.append(att.attrib['color'])

                    assert (len(car_per_frame) == len(speed_per_frame))

                    if len(boxes_per_frame) == 0:
                        boxes_labels_file.append(None)
                    else:
                        boxes_labels_file.append(boxes_per_frame)

                    if len(car_per_frame) == 0:
                        car_labels_file.append(None)
                    else:
                        car_labels_file.append(car_per_frame)

                    if len(speed_per_frame) == 0:
                        speed_labels_file.append(None)
                    else:
                        speed_labels_file.append(speed_per_frame)

                    if len(color_per_frame) == 0:
                        color_labels_file.append(None)
                    else:
                        color_labels_file.append(color_per_frame)

                    if len(intersection_per_frame) == 0:
                        intersection_labels_file.append(None)
                    else:
                        intersection_labels_file.append(intersection_per_frame)

                ## UPDATED: 2/19/2020 -- video_start_indices -> define the indices that video starts
                video_length = self.video_start_indices[i+1] - self.video_start_indices[i]
                if len(car_labels_file) < video_length:
                    initial_car_labels_length = len(car_labels_file)
                    car_labels_file.extend([None] * (video_length - initial_car_labels_length))
                    speed_labels_file.extend([None] * (video_length - len(speed_labels_file)))
                    intersection_labels_file.extend(
                        [None] * (video_length - len(intersection_labels_file)))
                    color_labels_file.extend([None] * (video_length - len(color_labels_file)))
                    boxes_labels_file.extend([None] * (video_length - len(boxes_labels_file)))
                    self.logger.debug(
                        f"FILE: {file} has been modified to match length added {video_length - initial_car_labels_length} more columns")
                    self.logger.debug(f"-->> {len(car_labels_file)}")
                    assert (len(car_labels_file) == video_length)
                elif len(car_labels_file) > video_length:
                    self.logger.error(
                        f"Length mismatch len(car_labels_file) {len(car_labels_file)}, self.video_start_indices[i] {self.video_start_indices[i+1]}")

                self.logger.debug("----------------")
                self.logger.debug(file)
                self.logger.debug(len(car_labels_file))
                self.logger.debug(self.video_start_indices[i + 1])


                car_labels.extend(car_labels_file)
                speed_labels.extend(speed_labels_file)
                intersection_labels.extend(intersection_labels_file)
                color_labels.extend(color_labels_file)
                boxes_dataset.extend(boxes_labels_file)

        self.boxes = boxes_dataset

        return [car_labels, speed_labels, color_labels, intersection_labels]




if __name__ == "__main__":
    import time

    st = time.time()
    loader = UADetracLoader()

    ## frame mismatch between xml and actual files in MVI_39811
    images = loader.load_cached_images()
    print(images.shape)
    video_start_indices = loader.get_video_start_indices()
    labels = loader.load_labels(dir = '/nethome/jbang36/eva/data/ua_detrac/DETRAC-Train-Annotations-XML')




    """
    images = loader.load_images()
    labels = loader.load_labels()
    boxes = loader.load_boxes()

    print("Time taken to load everything from disk", time.time() - st, "seconds")
    loader.save_boxes()
    loader.save_labels()
    loader.save_images()

    st = time.time()
    images_cached = loader.load_cached_images()
    labels_cached = loader.load_cached_labels()
    boxes_cached = loader.load_cached_boxes()
    print("Time taken to load everything from npy", time.time() - st, "seconds")

    assert (images.shape == images_cached.shape)
    assert (boxes.shape == boxes_cached.shape)

    for key, value in labels.items():
        assert(labels[key] == labels_cached[key])
    assert(labels.keys() == labels_cached.keys())
    
    """

