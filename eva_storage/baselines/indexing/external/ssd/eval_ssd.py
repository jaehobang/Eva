import torch
import os
from eva_storage.baselines.indexing.external.ssd.vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from eva_storage.baselines.indexing.external.ssd.vision.datasets.voc_dataset import VOCDataset
from eva_storage.baselines.indexing.external.ssd.vision.datasets.open_images import OpenImagesDataset
from eva_storage.baselines.indexing.external.ssd.vision.utils import measurements
from eva_storage.baselines.indexing.external.ssd.vision.utils.misc import str2bool
import argparse
import numpy as np
import logging
import sys

from eva_storage.baselines.indexing.external.ssd.vision.utils import box_utils



parser = argparse.ArgumentParser(description="SSD Evaluation on VOC Dataset.")
parser.add_argument('--net', default="vgg16-ssd",
                    help="The network architecture, it should be of mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')
parser.add_argument("--dataset", type=str, help="The root directory of the VOC dataset or Open Images dataset.")
parser.add_argument("--label_file", type=str, help="The label file path.")
parser.add_argument("--use_cuda", type=str2bool, default=True)
parser.add_argument("--use_2007_metric", type=str2bool, default=True)
parser.add_argument("--nms_method", type=str, default="hard")
parser.add_argument("--iou_threshold", type=float, default=0.5, help="The threshold of Intersection over Union.")
parser.add_argument("--eval_dir", default="eval_results", type=str, help="The directory to store custom_code results.")
parser.add_argument('--mb2_width_mult', default=1.0, type=float, help='Width Multiplifier for MobilenetV2')
args = parser.parse_args()
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")


def group_annotation_by_class(dataset):
    true_case_stat = {}
    all_gt_boxes = {}
    all_difficult_cases = {}
    for i in range(len(dataset)):
        image_id, annotation = dataset.get_annotation(i)
        gt_boxes, classes, is_difficult = annotation
        gt_boxes = torch.from_numpy(gt_boxes)
        for i, difficult in enumerate(is_difficult):
            class_index = int(classes[i])
            gt_box = gt_boxes[i]
            if not difficult:
                true_case_stat[class_index] = true_case_stat.get(class_index, 0) + 1

            if class_index not in all_gt_boxes:
                all_gt_boxes[class_index] = {}
            if image_id not in all_gt_boxes[class_index]:
                all_gt_boxes[class_index][image_id] = []
            all_gt_boxes[class_index][image_id].append(gt_box)
            if class_index not in all_difficult_cases:
                all_difficult_cases[class_index] = {}
            if image_id not in all_difficult_cases[class_index]:
                all_difficult_cases[class_index][image_id] = []
            all_difficult_cases[class_index][image_id].append(difficult)

    for class_index in all_gt_boxes:
        for image_id in all_gt_boxes[class_index]:
            all_gt_boxes[class_index][image_id] = torch.stack(all_gt_boxes[class_index][image_id])
    for class_index in all_difficult_cases:
        for image_id in all_difficult_cases[class_index]:
            all_gt_boxes[class_index][image_id] = torch.tensor(all_gt_boxes[class_index][image_id])
    return true_case_stat, all_gt_boxes, all_difficult_cases




def compute_average_precision_class_agnostic(num_true_casess, gt_boxess, difficult_casess, class_names, iou_threshold, use_2007_metric):
    import os
    eval_path = '/nethome/jbang36/eva/eva_storage/custom_code'

    final_true_positive = np.array([])
    final_false_positive = np.array([])


    for class_index, class_name in enumerate(class_names):

        if class_index == 0: continue #background

        print(class_index, class_name)
        prediction_file = os.path.join(eval_path, f"det_test_{class_name}.txt")
        num_true_cases = num_true_casess[class_index]
        gt_boxes = gt_boxess[class_index]
        difficult_cases = difficult_casess[class_index]

        ##### TODO: we can't just set false_positive[i] = 1, we have to do false_positive[i] += 1 because there can be multiple answers / mistakes in a given frame
        ##### TODO: I don't think VOC2007 measure took care of this because there is only one object per image....
        with open(prediction_file) as f:
            image_ids = []
            boxes = []
            scores = []
            for line in f:
                t = line.rstrip().split(" ")
                image_ids.append(t[0])
                scores.append(float(t[1]))
                box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
                box -= 1.0  # convert to python format where indexes start from 0
                boxes.append(box)
            scores = np.array(scores)
            sorted_indexes = np.argsort(-scores)
            boxes = [boxes[i] for i in sorted_indexes]
            image_ids = [image_ids[i] for i in sorted_indexes]
            true_positive = np.zeros(len(image_ids))
            false_positive = np.zeros(len(image_ids))
            matched = set()

            ### there are so many image ids that are not in gt_boxes.... this must be an error...
            #print(image_ids) ## this will return image ids in form of a string
            if type(image_ids[0]) == str:
                print("converting image ids to int instead of str")
                image_ids = list(map(int, image_ids))
                assert(type(image_ids[0]) == int)


            for i, image_id in enumerate(image_ids):

                box = boxes[i]
                if image_id not in gt_boxes:
                    false_positive[i] = 1
                    print(f"image_id {image_id} not in gt_boxes!!!! added {len(gt_boxes)} to false_positive array")
                    continue

                gt_box = gt_boxes[image_id]
                ious = box_utils.iou_of(box, gt_box)

                max_iou = torch.max(ious).item()
                max_arg = torch.argmax(ious).item()
                if max_iou > iou_threshold:
                    if difficult_cases[image_id][max_arg] == 0:
                        if (image_id, max_arg) not in matched:
                            true_positive[i] = 1

                            matched.add((image_id, max_arg))
                        else:

                            false_positive[i] = 1
                else:
                    false_positive[i] = 1
        final_true_positive = np.concatenate((final_true_positive, true_positive), axis = 0)
        final_false_positive = np.concatenate((final_false_positive, false_positive), axis = 0)

        final_true_positive = final_true_positive.cumsum()
        final_false_positive = final_false_positive.cumsum()
        precision = final_true_positive / (final_true_positive + final_false_positive)

        num_true = 0
        for key in num_true_casess.keys():
            num_true += num_true_casess[key]
        recall = final_true_positive / (num_true)
        """
        true_positive = true_positive.cumsum()
        false_positive = false_positive.cumsum()
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / num_true_cases
        """


        print("Printing stats for class...")
        print("true_positive", true_positive)
        print("false_positive", false_positive)
        print("precision is", precision)
        print("recall is", recall)
        if use_2007_metric:
            return measurements.compute_voc2007_average_precision(precision, recall)
        else:
            return measurements.compute_average_precision(precision, recall)




def compute_average_precision_per_class_practice(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    """
    This function is modified from compute_average_precision_per_class() by taking into account that multiple answers for each frame and multiple mistakes per each frame
    is accounted for....
    :param num_true_cases: number of true cases
    :param gt_boxes: number of ground truth boxes
    :param difficult_cases: whether it is a difficult case
    :param prediction_file: saved prediction file
    :param iou_threshold: iou_threshold needed to be considered a proposal box
    :param use_2007_metric: whether to use voc 2007 metric
    :return: average precision for a given class
    """
    with open(prediction_file) as f:
        ## you open the prediction files
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()

        ### there are so many image ids that are not in gt_boxes.... this must be an error...
        #print(image_ids) ## this will return image ids in form of a string
        if type(image_ids[0]) == str:
            print("converting image ids to int instead of str")
            image_ids = list(map(int, image_ids))
            assert(type(image_ids[0]) == int)


        for i, image_id in enumerate(image_ids):

            box = boxes[i]
            if image_id not in gt_boxes:
                ## how can this even happen? we filtered everything.....
                false_positive[i] += len(gt_boxes)
                #false_positive[i] = 1
                print("image_id", image_id, "not in gt_boxes!!!! skipping....")
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)

            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] += 1
                        #true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] += 1
                        #false_positive[i] = 1
            else:
                #false_positive[i] = 1
                false_positive[i] += 1

    print("before cum sum")
    print(len(true_positive))
    print(len(false_positive))
    print(true_positive)
    print(false_positive)
    print("---------------------")


    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    print("Printing stats for class...")
    print("true_positive", true_positive)
    print("false_positive", false_positive)
    print("precision is", precision)
    print("recall is", recall)
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)




def compute_average_precision_per_class_modified(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    """
    This function is modified from compute_average_precision_per_class() by taking into account that multiple answers for each frame and multiple mistakes per each frame
    is accounted for....
    :param num_true_cases: number of true cases
    :param gt_boxes: number of ground truth boxes
    :param difficult_cases: whether it is a difficult case
    :param prediction_file: saved prediction file
    :param iou_threshold: iou_threshold needed to be considered a proposal box
    :param use_2007_metric: whether to use voc 2007 metric
    :return: average precision for a given class
    """
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()

        ### there are so many image ids that are not in gt_boxes.... this must be an error...
        #print(image_ids) ## this will return image ids in form of a string
        if type(image_ids[0]) == str:
            print("converting image ids to int instead of str")
            image_ids = list(map(int, image_ids))
            assert(type(image_ids[0]) == int)


        for i, image_id in enumerate(image_ids):

            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] += len(gt_boxes)
                #false_positive[i] = 1
                print("image_id", image_id, "not in gt_boxes!!!! skipping....")
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)

            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] += 1
                        #true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] += 1
                        #false_positive[i] = 1
            else:
                #false_positive[i] = 1
                false_positive[i] += 1

    print("before cum sum")
    print(len(true_positive))
    print(len(false_positive))
    print(true_positive)
    print(false_positive)
    print("---------------------")


    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    print("Printing stats for class...")
    print("true_positive", true_positive)
    print("false_positive", false_positive)
    print("precision is", precision)
    print("recall is", recall)
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)



def compute_average_precision_per_class_weighed(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()

        ### there are so many image ids that are not in gt_boxes.... this must be an error...
        #print(image_ids) ## this will return image ids in form of a string
        if type(image_ids[0]) == str:
            print("converting image ids to int instead of str")
            image_ids = list(map(int, image_ids))
            assert(type(image_ids[0]) == int)


        for i, image_id in enumerate(image_ids):

            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                print("image_id", image_id, "not in gt_boxes!!!! skipping....")
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)

            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1


    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases

    print("Printing stats for class...")
    print("true_positive", true_positive)
    print("false_positive", false_positive)
    print("precision is", precision)
    print("recall is", recall)
    if use_2007_metric:
        return len(true_positive), measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return len(true_positive), measurements.compute_average_precision(precision, recall)



def compute_average_precision_per_class(num_true_cases, gt_boxes, difficult_cases,
                                        prediction_file, iou_threshold, use_2007_metric):
    with open(prediction_file) as f:
        image_ids = []
        boxes = []
        scores = []
        for line in f:
            t = line.rstrip().split(" ")
            image_ids.append(t[0])
            scores.append(float(t[1]))
            box = torch.tensor([float(v) for v in t[2:]]).unsqueeze(0)
            box -= 1.0  # convert to python format where indexes start from 0
            boxes.append(box)
        scores = np.array(scores)
        sorted_indexes = np.argsort(-scores)
        boxes = [boxes[i] for i in sorted_indexes]
        image_ids = [image_ids[i] for i in sorted_indexes]
        true_positive = np.zeros(len(image_ids))
        false_positive = np.zeros(len(image_ids))
        matched = set()

        ### there are so many image ids that are not in gt_boxes.... this must be an error...
        #print(image_ids) ## this will return image ids in form of a string
        if type(image_ids[0]) == str:
            print("converting image ids to int instead of str")
            image_ids = list(map(int, image_ids))
            assert(type(image_ids[0]) == int)


        for i, image_id in enumerate(image_ids):

            box = boxes[i]
            if image_id not in gt_boxes:
                false_positive[i] = 1
                print("image_id", image_id, "not in gt_boxes!!!! skipping....")
                continue

            gt_box = gt_boxes[image_id]
            ious = box_utils.iou_of(box, gt_box)

            max_iou = torch.max(ious).item()
            max_arg = torch.argmax(ious).item()
            if max_iou > iou_threshold:
                if difficult_cases[image_id][max_arg] == 0:
                    if (image_id, max_arg) not in matched:
                        true_positive[i] = 1
                        matched.add((image_id, max_arg))
                    else:
                        false_positive[i] = 1
            else:
                false_positive[i] = 1


    true_positive = true_positive.cumsum()
    false_positive = false_positive.cumsum()
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / num_true_cases



    print("Printing stats for class...")
    print("true_positive", true_positive)
    print("false_positive", false_positive)


    print("precision is", precision)
    print("recall is", recall)
    if use_2007_metric:
        return measurements.compute_voc2007_average_precision(precision, recall)
    else:
        return measurements.compute_average_precision(precision, recall)


if __name__ == '__main__':
    eval_path = pathlib.Path(args.eval_dir)
    eval_path.mkdir(exist_ok=True)
    timer = Timer()
    class_names = [name.strip() for name in open(args.label_file).readlines()]

    if args.dataset_type == "voc":
        dataset = VOCDataset(args.dataset, is_test=True)
    elif args.dataset_type == 'open_images':
        dataset = OpenImagesDataset(args.dataset, dataset_type="test")

    true_case_stat, all_gb_boxes, all_difficult_cases = group_annotation_by_class(dataset)
    if args.net == 'vgg16-ssd':
        net = create_vgg_ssd(len(class_names), is_test=True)


    timer.start("Load Model")
    net.load(args.trained_model)
    net = net.to(DEVICE)
    print(f'It took {timer.end("Load Model")} seconds to load the model.')
    if args.net == 'vgg16-ssd':
        predictor = create_vgg_ssd_predictor(net, nms_method=args.nms_method, device=DEVICE)


    results = []
    for i in range(len(dataset)):
        print("process image", i)
        timer.start("Load Image")
        image = dataset.get_image(i)
        print("Load Image: {:4f} seconds.".format(timer.end("Load Image")))
        timer.start("Predict")
        boxes, labels, probs = predictor.predict(image)
        print("Prediction: {:4f} seconds.".format(timer.end("Predict")))
        indexes = torch.ones(labels.size(0), 1, dtype=torch.float32) * i
        results.append(torch.cat([
            indexes.reshape(-1, 1),
            labels.reshape(-1, 1).float(),
            probs.reshape(-1, 1),
            boxes + 1.0  # matlab's indexes start from 1
        ], dim=1))
    results = torch.cat(results)
    for class_index, class_name in enumerate(class_names):
        if class_index == 0: continue  # ignore background
        prediction_path = os.path.join(eval_path, f"det_test_{class_name}.txt")
        with open(prediction_path, "w") as f:
            sub = results[results[:, 1] == class_index, :]
            for i in range(sub.size(0)):
                prob_box = sub[i, 2:].numpy()
                image_id = dataset.ids[int(sub[i, 0])]
                print(
                    image_id + " " + " ".join([str(v) for v in prob_box]),
                    file=f
                )
    aps = []
    print("\n\nAverage Precision Per-class:")
    for class_index, class_name in enumerate(class_names):
        if class_index == 0:
            continue
        prediction_path = eval_path / f"det_test_{class_name}.txt"
        ap = compute_average_precision_per_class(
            true_case_stat[class_index],
            all_gb_boxes[class_index],
            all_difficult_cases[class_index],
            prediction_path,
            args.iou_threshold,
            args.use_2007_metric
        )
        aps.append(ap)
        print(f"{class_name}: {ap}")

    print(f"\nAverage Precision Across All Classes:{sum(aps)/len(aps)}")



