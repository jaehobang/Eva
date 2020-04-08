

from loaders.uadetrac_loader import UADetracLoader
from filters.minimum_filter import FilterMinimum
from utils import Utils
from eva_storage.baselines.indexing.external.ssd.custom_code.ssd_predictor_wrapper import SSD_predictor_wrapper



def remap_indices(potential_frame_indices, final_potential_frame_indices):
    udf_i = 0
    new_list = potential_frame_indices.copy()
    for filter_i, value in enumerate(potential_frame_indices):
        if value == 1:
            new_list[filter_i] = final_potential_frame_indices[udf_i]
            udf_i += 1
    return new_list



if __name__ == "__main__":

    ### Let's assume you are interested in running a query such as `select FRAMES where FRAME contains VAN`

    ### load the UADetrac dataset
    loaders = UADetracLoader()
    images = loaders.load_cached_images()
    labels = loaders.load_cached_labels()
    boxes = loaders.load_cached_boxes()
    ### generate binary labels for filters
    utils_eva = Utils()
    binary_labels = utils_eva.labels2binaryUAD(labels, ['van']) ## can supply as ['car', 'van', 'bus', 'others'] to get binary labels for all


    ### train the filter -- later we can look into options that save these models
    ##### https://scikit-learn.org/stable/modules/model_persistence.html
    filters = FilterMinimum()
    filters.train(images, binary_labels)

    ### train the UDF
    ### NOTE: to train the UDF (SSD), it is much more convenient to run `python train_ssd_uad.py` -- I organized the code so that it runs out of the box and trains an SSD model that can be used for evaluation
    ###       Hence, let's assume you ran this code and have a trained SSD model

    ###############################################
    ######## Pipeline Execution ###################
    ###############################################

    ### Since we have already loaded UADetrac, we will skip this step

    ### create an instance of predictor -- I made this custom script but haven't tested it, so you probably would need to make some modifications
    class_names = ['BACKGROUND', 'car', 'bus', 'others', 'van']
    model_path = 'path/to/your/saved/model'
    predictor = SSD_predictor_wrapper(model_path, class_names)


    potential_frame_indices = filters.predict(images, post_model_name='rf')
    potential_images = images[potential_frame_indices]
    final_potential_image_indices = predictor.predict(potential_images, ['van'])
    final_potential_image_indices = final_potential_image_indices['van']

    ### now we have to map the indices back to the original image indices
    final_indices = remap_indices(potential_frame_indices, final_potential_image_indices)










