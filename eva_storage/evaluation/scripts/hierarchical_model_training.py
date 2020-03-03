


def train_conservative(train_images, preprocess, name):
    segmented_images = preprocess.run(train_images, None, load=True)
    model = UNet()
    model.train(train_images, segmented_images, save_name = name)
    return



if __name__ == "__main__":
    # %%
    import os
    import time


    from loaders.uadetrac_loader import UADetracLoader
    from eva_storage.preprocessingModule import PreprocessingModule
    from eva_storage.UNet import UNet
    from eva_storage.postprocessingModule import PostprocessingModule
    from eva_storage.logger import LoggingLevel, Logger

    loader = UADetracLoader()
    preprocess = PreprocessingModule()
    bare_model = UNet()
    postprocess = PostprocessingModule()
    logger = Logger()



    st = time.time()
    # 1. Load the images (cached images is fine)
    images = loader.load_cached_images()
    labels = loader.load_cached_labels()
    video_start_indices = loader.get_video_start_indices()
    print(f"Done loading images in {time.time() - st} (sec)")



    directory_begin = '/nethome/jbang36/eva_jaeho/data/models/'
    """
    model_names = ['history20_dist2thresh300',
                   'history20_dist2thresh300_bloat_lvl2',
                   'history20_dist2thresh300_bloat_lvl3',
                   'history20_dist2thresh300_bloat_lvl4']
    
    model_names = ['history20_dist2thresh300_bloat_lvl2',
                   'history20_dist2thresh300_bloat_lvl3',
                   'history20_dist2thresh300_bloat_lvl4']
    """

    ## this is needed because we only have segmented images and don't have the primary network saved
    train_conservative(images, preprocess, 'history2_dist2thresh300')


    model_names = ['history2_dist2thresh300',
                   'history2_dist2thresh300_alt_lvl2',
                   'history2_dist2thresh300_alt_lvl3',
                   'history2_dist2thresh300_alt_lvl4']



    ## train the models
    for i in range(3):
        logger.info("--------------------------------------")
        logger.info(f"Starting level {i+2} training...")
        load_dir = os.path.join(directory_begin, model_names[i]+'-epoch60.pth')
        logger.info(f"initial model loading directory is {load_dir}")

        _, level_output = bare_model.execute(images, load_dir = load_dir)

        if i % 2 == 1:
            level_output = postprocess.run_bloat(level_output) ## we will keep running bloat methods per iteration and see what happens
        else:
            level_output = postprocess.run_erosion(level_output)  ## we will keep running bloat methods per iteration and see what happens

        level_model = UNet()
        level_model.train(images, level_output, model_names[i+1])

