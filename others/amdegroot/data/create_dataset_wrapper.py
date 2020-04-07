import os

from others.amdegroot.data.coco import COCO_ROOT, COCODetection
from others.amdegroot.data.voc0712 import VOC_ROOT, VOCDetection
from others.amdegroot.data.uad import UAD_ROOT, UADDetection
from others.amdegroot.utils.augmentations import SSDAugmentation
from others.amdegroot.data.config import *
from loaders.uadetrac_loader import UADetracLoader
from logger import Logger



available_datasets = {'COCO': COCO_ROOT
                        , 'VOC': VOC_ROOT
                        , 'UAD': UAD_ROOT
                        , 'JNET': UAD_ROOT}


logger = Logger()



def create_dataset(dataset_name, is_train=None, cache_name = None):
    if dataset_name not in available_datasets.keys():
        logger.error(f"dataset: {dataset_name} not in {vailable_datasets.keys()}")

    if (dataset_name is 'UAD' or dataset_name is 'JNET') and (is_train is None):
        logger.error(f"Must specify training or testing for UAD and JNET!")

    if dataset_name == 'COCO':
        if not os.path.exists(COCO_ROOT):
            logger.error('Must specify dataset_root if specifying dataset')
        cfg = coco
        dataset = COCODetection(root=COCO_ROOT,
                                transform=SSDAugmentation(cfg['min_dim'],
                                                          MEANS))
    elif dataset_name == 'VOC':
        cfg = voc
        dataset = VOCDetection(root=VOC_ROOT,
                               transform=SSDAugmentation(cfg['min_dim'],
                                                         MEANS))

    elif dataset_name == 'UAD' and is_train:
        logger.info("We are loading UADetrac!!")
        cfg = uad
        dataset = UADDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
        loader = UADetracLoader()
        images = loader.load_cached_images(name='uad_train_images.npy', vi_name = 'uad_train_vi.npy' )
        boxes = loader.load_cached_boxes(name = 'uad_train_boxes.npy')
        labels = loader.load_cached_labels(name = 'uad_train_labels.npy')

        labels = labels['vehicle']
        images, labels, boxes = loader.filter_input3(images, labels, boxes)
        dataset.set_images(images)
        dataset.set_labels(labels)
        dataset.set_boxes(boxes)

    elif dataset_name == 'UAD' and not is_train:
        logger.info("We are loading UADetrac!!")
        cfg = uad
        dataset = UADDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
        loader = UADetracLoader()
        images = loader.load_cached_images(name = 'uad_test_images.npy', vi_name = 'uad_test_vi.npy')
        boxes = loader.load_cached_boxes(name = 'uad_test_boxes.npy')
        labels = loader.load_cached_labels(name = 'uad_test_labels.npy')
        labels = labels['vehicle']
        images, labels, boxes = loader.filter_input3(images, labels, boxes)
        images = images[:4000]
        labels = labels[:4000]
        boxes = boxes[:4000]

        dataset.set_images(images)
        dataset.set_labels(labels)
        dataset.set_boxes(boxes)

    elif dataset_name == 'JNET' and is_train:
        if cache_name is None:
            logger.error("Cache name is required for JNET!! returning...")
            return None
        logger.info("We are loading JNET - UADetrac!!")
        cfg = uad
        dataset = UADDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
        loader = UADetracLoader()
        images = loader.load_cached_images(name=cache_name, vi_name = 'uad_train_vi.npy')
        labels = loader.load_cached_labels(name='uad_train_labels.npy')
        boxes = loader.load_cached_boxes(name='uad_train_boxes.npy')
        labels = labels['vehicle']

        images, labels, boxes = loader.filter_input3(images, labels, boxes)

        logger.info(f"images shape is {images.shape}")
        logger.info(f"labels length is {len(labels)}")
        logger.info(f"boxes length is {len(boxes)}")
        assert(images.shape[0] == len(labels))
        assert(len(labels) == len(boxes))
        dataset.set_images(images)
        dataset.set_labels(labels)
        dataset.set_boxes(boxes)

    elif dataset_name == 'JNET' and not is_train:
        if cache_name is None:
            logger.error("Cache name is required for JNET! returning....")
            return
        logger.info("We are loading JNET - UADetrac!!")
        cfg = uad
        dataset = UADDetection(transform=SSDAugmentation(cfg['min_dim'], MEANS))
        loader = UADetracLoader()
        images = loader.load_cached_images(name = cache_name, vi_name = 'uad_test_vi.npy')
        labels = loader.load_cached_labels(name='uad_test_labels.npy')
        boxes = loader.load_cached_boxes(name = 'uad_test_boxes.npy')
        labels = labels['vehicle']
        images, labels, boxes = loader.filter_input3(images, labels, boxes)

        ###FIXED: we will make this really small so that numbers appear fast
        images = images[:2000]
        labels = labels[:2000]
        boxes = boxes[:2000]

        logger.info(f"images shape is {images.shape}")
        logger.info(f"labels length is {len(labels)}")
        logger.info(f"boxes length is {len(boxes)}")
        dataset.set_images(images)
        dataset.set_labels(labels)
        dataset.set_boxes(boxes)

    return dataset, cfg



if __name__ == "__main__":
    ### let's save some things
    ### save the images, labels, boxes for all test and train
    logger.info("starting.....")
    loader = UADetracLoader()
    """
    images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/4_images')
    labels, boxes = loader.load_labels('/nethome/jbang36/eva_jaeho/data/ua_detrac/4_xml')
    assert(len(images) == len(boxes))
    loader.save_images(name = 'uad_train_images.npy', vi_name='uad_train_vi.npy')
    loader.save_labels(name = 'uad_train_labels.npy')
    loader.save_boxes(name = 'uad_train_boxes.npy')

    logger.info("Saved all train data!")
    """
    test_images = loader.load_images(dir='/nethome/jbang36/eva_jaeho/data/ua_detrac/5_images')
    test_labels, test_boxes = loader.load_labels('/nethome/jbang36/eva_jaeho/data/ua_detrac/5_xml')
    assert(len(test_images) == len(test_boxes))

    loader.save_images(name='uad_test_images.npy', vi_name='uad_test_vi.npy')
    loader.save_labels(name='uad_test_labels.npy')
    loader.save_boxes(name='uad_test_boxes.npy')

    logger.info("Saved all test data!")
