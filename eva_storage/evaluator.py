"""
This file will contain all the evaluation functions
"""

from eva_storage.logger import LoggingLevel, Logger


class Evaluator:

    def __init__(self):
        self.logger = Logger()


    def debugMode(self, mode = False):
        if mode:
            self.logger.setLogLevel(LoggingLevel.DEBUG)
        else:
            self.logger.setLogLevel(LoggingLevel.INFO)




    # ground_truth_list: list of ground truth boxes where each element refers to all the boxes existing in the frame
    # proposed_list: list of proposed boxes where each element refers to all the boxes existing in the frame
    # filter: whether to give some filtering arguments such as minimum size / aspect ratio
    # iou: minimum area of overlap needed to be considered a true positive
    def corloc(self, gt_boxes, proposed_boxes, filter=False, iou=0.5):
        total_count = 0
        tp_count = 0
        fn_count = 0
        fp_count = 0
        if filter:
            gt_boxes = self.filter_ground_truth(gt_boxes)

        # number of frames involved should be the same
        assert (len(gt_boxes) == len(proposed_boxes))

        for i in range(len(gt_boxes)):
            tp, fp, fn = self.compute_overlap(gt_boxes[i], proposed_boxes[i], iou)
            tp_count += tp
            fn_count += fn
            fp_count += fp

        # compute precision and recall
        if tp_count + fp_count == 0:
            precision = 0
        else:
            precision = tp_count / (tp_count + fp_count)

        if tp_count + fn_count == 0:
            recall = 0
        else:
            recall = tp_count / (tp_count + fn_count)

        ## debugging
        print("true positive", tp_count)
        print("false positive", fp_count)
        print("false negative", fn_count)

        return precision, recall


    def filter_ground_truth(self, ground_truth_list, input_patch_type='ml', output_patch_type='ml'):
        new_ground_truth_list = []
        if input_patch_type == 'ml':
            tmp = []
            for i in range(len(ground_truth_list)):
                tmp.append(self.ml2cv_patches(ground_truth_list[i]))
            ground_truth_list = tmp

        ## filter patches takes in cv type patches so make sure to convert beforehand!
        for patches_frame in ground_truth_list:
            new_ground_truth_list.append(self.filter_patches(patches_frame))
        assert (len(new_ground_truth_list) == len(ground_truth_list))

        if output_patch_type == 'ml':
            tmp = []
            for i in range(len(new_ground_truth_list)):
                tmp.append(self.cv2ml_patches(new_ground_truth_list[i]))
            new_ground_truth_list = tmp

        return new_ground_truth_list


    def ml2cv_patches(self, ml_patches):
        # cv convention is (left, top, width, height)
        if ml_patches == None:
            return None
        cv_patches = []
        for patch in ml_patches:
            top, left, bottom, right = patch
            cv_patch = (left, top, right - left, bottom - top)
            cv_patches.append(cv_patch)
        return cv_patches


    def cv2ml_patches(self, cv_patches):
        if cv_patches == None:
            return None
        ml_patches = []
        for patch in cv_patches:
            left, top, width, height = patch
            ml_patch = (top, left, top + height, left + width)
            ml_patches.append(ml_patch)
        return ml_patches


    def filter_patches(self, patches, img_height=300, img_width=300,
                       min_ratio_image=0.05, max_ratio_image=0.7,
                       min_ratio_patch=0.5, max_ratio_patch=3.0):
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


    def compute_overlap(self, ground_truth_frame, proposed_list_frame, iou=0.5):
        # first work the naive version -> do the multithreaded version
        tp_count = 0
        fn_count = 0
        fp_count = 0
        if ground_truth_frame == None and proposed_list_frame == None:
            tp_count = 0
            fn_count = 0
            fp_count = 0
        elif ground_truth_frame == None:
            # there should not be boxes but model detects boxes
            # these are false positive
            fp_count = len(proposed_list_frame)

        elif proposed_list_frame == None:
            # there should be boxes but the model detects none
            # these are false negatives
            fn_count = len(ground_truth_frame)
        else:
            fp_list = [True] * len(proposed_list_frame)
            seen_proposed_boxes = [False] * len(proposed_list_frame)
            for ground_box in ground_truth_frame:
                matching_box = False
                for ii, proposed_box in enumerate(proposed_list_frame):
                    if seen_proposed_boxes[ii]:
                        # if we have already seen this box and computed into the results,
                        # don't recompute this into calculation
                        continue

                    overlap_area = 0
                    p_top, p_left, p_bottom, p_right = proposed_box
                    g_top, g_left, g_bottom, g_right = ground_box

                    # determine the actual area of overlap
                    # for overlap ground_box needs to intersect proposed_box in at least two ways (vertical and horizontal)
                    #
                    i_top = max(p_top, g_top)
                    i_left = max(p_left, g_left)
                    i_bottom = min(p_bottom, g_bottom)
                    i_right = min(p_right, g_right)
                    # check if left is really left, top is really top
                    if i_left < i_right and i_top < i_bottom:  # else there is no overlap
                        overlap_area = (i_right - i_left) * (i_bottom - i_top)

                    p_area = (p_right - p_left) * (p_bottom - p_top)
                    g_area = (g_right - g_left) * (g_bottom - g_top)
                    if overlap_area / (p_area + g_area - overlap_area) >= iou:
                        tp_count += 1
                        matching_box = True
                        fp_list[ii] = False
                        seen_proposed_boxes[ii] = True
                if matching_box == False:
                    fn_count += 1
            fp_count = sum(fp_list)
        return tp_count, fp_count, fn_count
