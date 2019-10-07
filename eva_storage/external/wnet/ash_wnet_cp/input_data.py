import tensorflow as tf
import os
import numpy as np

file_path = ""
batch_size = 1

def parse_image(filename):
    image_string = tf.read_file(file_path + filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])
    img_standard = tf.image.per_image_standardization(image_resized)
    return img_standard


def get_filenames():
    filenames = os.listdir(file_path)
    return filenames


def get_filenames_uadetrac():
    filenames = []
    dir = "/nethome/jbang36/eva/data/ua_detrac/small-data"
    for root, subdirs, files in os.walk(dir):
        files.sort()
        for file in files:
            filenames.append(os.path.join(root, file))
    return filenames


def input_data_uadetrac():
    filenames = get_filenames_uadetrac()
    print(len(filenames))
    train_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    train_dataset = train_dataset.shuffle(100).repeat()
    train_dataset = train_dataset.map(parse_image, num_parallel_calls=4).batch(batch_size)
    return train_dataset.make_one_shot_iterator()


def input_data():
    filenames = get_filenames()
    print(len(filenames))
    train_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    train_dataset = train_dataset.shuffle(100).repeat()
    train_dataset = train_dataset.map(parse_image, num_parallel_calls=4).batch(batch_size)
    return train_dataset.make_one_shot_iterator()

if __name__ == '__main__':
    iterator = input_data()
    images = iterator.get_next()
    print(images)

