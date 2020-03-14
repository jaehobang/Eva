"""
This file is used to evaluate the compression method of the pipeline
@Jaeho Bang
"""

#import history.helpers as helpers
from eva_storage.evaluation.old.evaluate_compression import *





def get_rep_frames(images:np.ndarray, labels, image_cluster_labels):
    visited_cluster_nums = set()
    n_samples, height, width ,channels = images.shape
    rep_images = np.zeros(shape = (max(image_cluster_labels) + 1, height, width, channels))
    rep_labels = np.zeros(shape = (max(image_cluster_labels) + 1))

    for i in range(len(image_cluster_labels)):
        if image_cluster_labels[i] not in visited_cluster_nums:
            visited_cluster_nums.add(image_cluster_labels[i])
            rep_images[image_cluster_labels[i]] = images[i]
            rep_labels[image_cluster_labels[i]] = labels[i]

    return rep_images, rep_labels



### ipynb files such as evaluate_compression and evaluation_compression_dl will use this file.

def get_rep_frames():

    loader = UADetracLoader()
    images = loader.load_cached_images()
    labels = loader.load_cached_labels()
    video_start_indices = loader.get_video_start_indices()
    print("Done loading data")

    pm = PreprocessingModule()
    seg_images = pm.run(images, video_start_indices)
    print("Done generating primary masks")

    unet = UNet()
    unet.train(images, seg_images, epoch=100)
    print("Done training unet")

    unet_compressed_images, unet_segmented_images = unet.execute()
    cm = ClusterModule()
    image_cluster_labels = cm.run(unet_compressed_images)
    print("Done clustering")

    # Generate binary labels
    ## within labels['vehicle'] there are ['car', 'others', 'van', 'bus']

    car_labels = helpers.generateBinaryLabels(labels['vehicle'])
    other_labels = helpers.generateBinaryLabels(labels['vehicle'], label_of_interest='others')
    van_labels = helpers.generateBinaryLabels(labels['vehicle'], 'van')
    bus_labels = helpers.generateBinaryLabels(labels['vehicle'], 'bus')

    ## divide the training and validation dataset for rep frames
    division_point = int(rep_images.shape[0] * 0.8)

    rep_train_set = {}
    rep_test_set = {}
    for key in ['van', 'bus']:
        rep_train_set[key] = get_rep_frames(train_set[key][0], train_set[key][1], train_set[key][2])
        rep_test_set[key] = get_rep_frames(test_set[key][0], test_set[key][1], test_set[key][2])

    return




if __name__ == "__main__":

    ### deprecated... moved to ipynb file
    """
    loader = LoaderUADetrac()
    images = loader.load_cached_images()
    labels = loader.load_cached_labels()
    video_start_indices = loader.get_video_start_indices()
    pm = PreprocessingModule()
    seg_images = pm.run(images,video_start_indices)
    unet = UNet()
    unet.train(images, seg_images)
    unet_compressed_images, unet_segmented_images = unet.execute()
    cm = ClusterModule()
    image_cluster_labels = cm.run(unet_compressed_images)

    rep_images, rep_labels = get_rep_frames(images, labels['vehicle'], image_cluster_labels)
    ## TODO: Chose the representative frames... now need to do custom_code with filters




    # init a filter instance that is trained on all images
    fm_everyframe = FilterMinimum()
    fm_everyframe.train(images, labels['vehicle'])

    fm_repframe = FilterMinimum()
    fm_repframe.train(rep_images, rep_labels)
    """

    


