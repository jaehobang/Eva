import os
import torch
import time
import numpy as np
from loaders.uadetrac_loader import UADetracLoader
from logger import Logger
from eva_storage.baselines.indexing.external.ssd.custom_code.train_jnet_ssd_uad import load_jnet_results
import eva_storage.baselines.indexing.external.ssd.custom_code.util_ssd_uad as util_custom



if __name__ == "__main__":
    ## variables we need to change
    save_file_output = 'jnet_test-200-300.npy'
    model_directory = '/nethome/jbang36/eva_jaeho/data/models/history200_dist2thresh300-epoch60.pth'


    os.environ["CUDA_VISIBLE_DEVICES"] = "1" ## we want to run everything on gpu 1
    DEVICE = train_device = torch.device('cuda') ## will this execute everything on gpu 1?
    loader = UADetracLoader()
    logger = Logger()

    tic = time.perf_counter()

    ## we load from directories given
    images = loader.load_cached_images(name = 'uad_test_images.npy', vi_name = 'uad_test_vi.npy')
    ## before we divide the dataset, we want to load the segmented images..or process them
    save_directory = os.path.join('/nethome/jbang36/eva_jaeho/data/npy_files', save_file_output)

    logger.info(f"Starting segmentation with model in {model_directory}")
    tic = time.perf_counter()
    images = load_jnet_results(images, model_directory= model_directory)

    np.save(save_directory, images)

    logger.info(f"Saved the results to {save_directory}")
