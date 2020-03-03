"""
This module performs the postprocessing step.
Performs computer vision techniques on the output of learning models
If any issues arise please contact jaeho.bang@gmail.com

@Jaeho Bang

"""

import numpy as np
import cv2
import os
import config
import time
from loaders.uadetrac_loader import UADetracLoader
from eva_storage.logger import Logger, LoggingLevel


TIMED = True


class PostprocessingModule:

    def __init__(self):
        self.postprocessed_images = None
        self.logger = Logger()
        self.filtered_boxes = []
        self.unfiltered_boxes = []


    def debugMode(self, mode = False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)



    def run(self, images:np.ndarray, load = False):
        """
        Try loading the data
        If there is nothing to load, we have to manually go through the process

        :param images:
        :param video_start_indices:
        :return:
        """

        st = time.perf_counter()
        self.postprocessed_images = None

        if images.ndim != 3:
            self.logger.error(f"Expected images to have squeezed depth channel, but shape of matrix is {images.shape}")
            raise ValueError

        if images.dtype != np.uint8:
            self.logger.error(f"Expected images to be of dtype np.uint8 but got {images.dtype}")
            raise ValueError


        self.logger.info("Starting CV functionalities on segmented images...")
        if load:
            self.logger.info("Trying to load from saved file....")
            self._loadPostprocessedImages()



        if self.postprocessed_images is None:

            postprocessed_images = np.ndarray(shape=images.shape)
            kernel = np.ones((3, 3), np.uint8)

            # fgbg only takes grayscale images, we need to convert
            ## check if image is already converted to grayscale -> channels = 1

            for i in range(images.shape[0]):
                image = images[i]

                seg_cp = np.copy(image)
                med = cv2.medianBlur(seg_cp, 5)
                ret, otsu_m = cv2.threshold(med, 0, 255, cv2.THRESH_OTSU)
                postprocessed_images[i] = otsu_m


            self.postprocessed_images = postprocessed_images.astype(np.uint8)


        self.logger.info(f"Done in {time.perf_counter() - st} (sec)")
        return self.postprocessed_images


    def run_erosion(self, images:np.ndarray, load = False):
        """
        Try loading the data
        If there is nothing to load, we have to manually go through the process

        :param images:
        :param video_start_indices:
        :return:
        """

        st = time.perf_counter()
        self.postprocessed_images = None

        if images.ndim != 3:
            self.logger.error(f"Expected images to have squeezed depth channel, but shape of matrix is {images.shape}")
            raise ValueError

        if images.dtype != np.uint8:
            self.logger.error(f"Expected images to be of dtype np.uint8 but got {images.dtype}")
            raise ValueError

        self.logger.info("Starting CV functionalities on segmented images...")
        if load:
            self.logger.info("Trying to load from saved file....")
            self._loadPostprocessedImages()

        if self.postprocessed_images is None:

            postprocessed_images = np.ndarray(shape=images.shape)
            kernel = np.ones((3, 3), np.uint8)

            # fgbg only takes grayscale images, we need to convert
            ## check if image is already converted to grayscale -> channels = 1

            for i in range(images.shape[0]):
                image = images[i]

                seg_cp = np.copy(image)

                med = cv2.medianBlur(seg_cp, 5)
                erosion = cv2.erode(med, kernel, iterations=3)
                ret, otsu_m = cv2.threshold(erosion, 0, 255, cv2.THRESH_OTSU)
                postprocessed_images[i] = otsu_m

            self.postprocessed_images = postprocessed_images.astype(np.uint8)

        self.logger.info(f"Done in {time.perf_counter() - st} (sec)")
        return self.postprocessed_images



    def run_bloat(self, images:np.ndarray, load = False):
        """
        Try loading the data
        If there is nothing to load, we have to manually go through the process

        :param images:
        :param video_start_indices:
        :return:
        """

        st = time.perf_counter()
        self.postprocessed_images = None

        if images.ndim != 3:
            self.logger.error(f"Expected images to have squeezed depth channel, but shape of matrix is {images.shape}")
            raise ValueError

        if images.dtype != np.uint8:
            self.logger.error(f"Expected images to be of dtype np.uint8 but got {images.dtype}")
            raise ValueError


        self.logger.info("Starting CV functionalities on segmented images...")
        if load:
            self.logger.info("Trying to load from saved file....")
            self._loadPostprocessedImages()



        if self.postprocessed_images is None:

            postprocessed_images = np.ndarray(shape=images.shape)
            kernel = np.ones((3, 3), np.uint8)

            # fgbg only takes grayscale images, we need to convert
            ## check if image is already converted to grayscale -> channels = 1

            for i in range(images.shape[0]):
                image = images[i]

                seg_cp = np.copy(image)
                med = cv2.medianBlur(seg_cp, 5)
                dilation = cv2.dilate(med, kernel, iterations=3)
                ret, otsu_m = cv2.threshold(dilation, 0, 255, cv2.THRESH_OTSU)
                postprocessed_images[i] = otsu_m


            self.postprocessed_images = postprocessed_images.astype(np.uint8)


        self.logger.info(f"Done in {time.perf_counter() - st} (sec)")
        return self.postprocessed_images



    def detectBoxes(self, postprocessed_images):
        """

        :param postprocessed_images: shape = [n_samples, height, width, channels]
        :return: list of boxes in evaluation format
        """
        self.unfiltered_boxes = []
        self.filtered_boxes = []

        st = time.perf_counter()
        self.logger.debug(f"Processing {len(postprocessed_images)}...")
        for i in range(len(postprocessed_images)):
            image = postprocessed_images[i]

            patches = self.detect_patches(image)
            new_patches = self.filter_patches(patches)
            self.logger.debug(f"Number of unfiltered boxes {len(patches)}, Number of filtered boxes {len(new_patches)}")
            filtered_patches =self.format_patches(new_patches)
            unfiltered_patches = self.format_patches(patches)

            self.filtered_boxes.append(filtered_patches)
            self.unfiltered_boxes.append(unfiltered_patches)

        self.logger.info(f"Done with box detection in {time.perf_counter() - st} (sec)")
        return self.unfiltered_boxes, self.filtered_boxes

    def format_patches(self, cv_box_format):
        """
        The reason we have to format patches in the first place is becausecv outputs coordinates as follows: [left, top, width, height]
        However, nobody does this.... we will use the format: [left, top, right, bottom]
        :param patch:
        :return:
        """

        ## we will assume patches are in a 2d list
        conventional_box_format = []
        for i in range(len(cv_box_format)):
            left, top, width, height = cv_box_format[i]
            conventional_box_format.append([left, top, left+width, top+height])
        assert(len(conventional_box_format) == len(cv_box_format))
        return conventional_box_format



    def filter_patches(self, patches, img_height=300, img_width=300,
                       min_ratio_image=0.05, max_ratio_image=0.7,
                       min_ratio_patch=0.5, max_ratio_patch=3.0):

        """

        :param img_height: height of image
        :param img_width: width of image
        :param min_ratio_image: ratio of patch to image (both width and height need to satisfy this constraint)
        :param max_ratio_image: ratio of patch to image (both width and height need to satisfy this constraint)
        :param min_ratio_patch: ratio of patch_height to patch_width
        :param max_ratio_patch: ratio of patch_height to patch_width
        :return: final boxes(patches)
        """
        # we want to filter all the patches and return new patches that satisfy the constraint
        new_patches = []

        if patches == None:
            return None
        for patch in patches:
            patch_height = patch[3]
            patch_width = patch[2]
            height_ratio = patch_height / img_height
            width_ratio = patch_width / img_width
            ratio_patch = patch_height / patch_width
            if height_ratio >= min_ratio_image and height_ratio <= max_ratio_image and \
                    width_ratio >= min_ratio_image and width_ratio <= max_ratio_image and \
                    ratio_patch >= min_ratio_patch and ratio_patch <= max_ratio_patch:
                new_patches.append(patch)
        return new_patches


    def detect_patches(self, image, shape='rectangle', mode=2):
        """
        This function detects the contours in segmented images
        :param shape: desired detected shaped
        :param mode:
        :return: boundRect[i][0] - left, boundRect[i][1] - top, boundRect[i][2] - width, boundRect[i][3] - height
        """
        if mode == 1:
            _, contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        elif mode == 2:
            _, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #contours = contours[0]


        if len(contours) == 0:
            print("No contours in image")
            return None

        if shape != 'rectangle':
            print("Method not support")
            return None
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])

        # boundRect[i][0] - left, boundRect[i][1] - top, boundRect[i][2] - width, boundRect[i][3] - height
        return boundRect


    def savePostprocessedImages(self, overwrite = False):
        eva_dir = config.eva_dir
        directory = os.path.join(eva_dir, 'data', 'npy_files', 'postprocessed_images.npy')
        if self.postprocessed_images is None:
            self.logger.error("Post processed images not available")
            raise ValueError
        elif os.path.exists(directory) and overwrite == False:
            self.logger.error("Same filename exists.. to overwrite, make sure the overwrite option is True")
            raise ValueError

        else:
            np.save(directory, self.segmented_images)
            self.logger.info(f"Saved segmented images to {directory}")


    def _loadPostprocessedImages(self, images):
        eva_dir = config.eva_dir
        dir = os.path.join(eva_dir, 'data', 'npy_files', 'postprocessed_images.npy')
        if os.path.exists(dir):
            self.logger.info("path", dir, "found!")
            self.segmented_images = np.load(dir)
            if self.segmented_images.shape != images.shape:
                self.segmented_images = None

        else:
            self.logger.error(f"path: {dir} does not exist...")


if __name__ == "__main__":
    logger = Logger()
    logger.info("logger instantiated...")

    logger.debug("you should not see this")
    logger.setLogLevel(LoggingLevel.DEBUG)
    logger.debug("you should see this")

    """
    loader = UADetracLoader()
    images = loader.load_images()
    labels = loader.load_labels()
    boxes = loader.load_boxes()
    video_start_indices = loader.get_video_start_indices()
    #images loaded as 300x300 - prepare the images
    preprocessor = PreprocessingModule()
    segmented_images = preprocessor.run(images, video_start_indices)
    """


