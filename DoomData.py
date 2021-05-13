import os

import numpy as np
from cv2 import cv2
import tensorflow as tf


class DoomData:

    def __init__(self):
        self.img_height = 128
        self.img_width = 128

    def create_dataset(self, img_folder):
        img_data_array = []
        class_name = []

        for dir1 in os.listdir(img_folder):
            for file in os.listdir(os.path.join(img_folder, dir1)):
                image_path = os.path.join(img_folder, dir1, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.img_height, self.img_width), interpolation=cv2.INTER_NEAREST)
                image = np.array(image)
                image = image.astype('float32')
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array, class_name

    def get_trainable_data(self, data_dir):
        img_data, class_name = self.create_dataset(data_dir)
        img_data_array = np.array(img_data)

        train_images = img_data_array.reshape((img_data_array.shape[0], 128, 128, 1)).astype('float32')

        BUFFER_SIZE = 60000
        BATCH_SIZE = 32

        train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        return train_images
