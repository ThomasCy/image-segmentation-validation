import os
from config import train_config as config
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image

resized_image_size = config["resized_image_size"]

# Splits the images into train, dev and test set


def split_photos(data_path):
    x = [data_path + i for i in sorted(os.listdir(data_path))]
    x_train, x_dev = train_test_split(
        x, test_size=config["train_dev+test_split"], random_state=0)
    x_dev, x_test = train_test_split(x_dev, test_size=0.1, random_state=0)
    return x_train, x_dev, x_test


def load_dataset(data_path):
    x_train, x_dev, x_test = split_photos(data_path)
    x_train_images = np.empty(
        (len(x_train), resized_image_size, resized_image_size, 3))
    x_dev_images = np.empty(
        (len(x_dev), resized_image_size, resized_image_size, 3))
    x_test_images = np.empty(
        (len(x_test), resized_image_size, resized_image_size, 3))

    for index, image_path in enumerate(x_train):
        image_data = Image.open(image_path)
        image_data = image_data.resize(
            (resized_image_size, resized_image_size))
        x_train_images[index] = np.array(image_data) / 255.0
    for index, image_path in enumerate(x_dev):
        image_data = Image.open(image_path)
        image_data = image_data.resize(
            (resized_image_size, resized_image_size))
        x_dev_images[index] = np.array(image_data) / 255.0
    for index, image_path in enumerate(x_test):
        image_data = Image.open(image_path)
        image_data = image_data.resize(
            (resized_image_size, resized_image_size))
        x_test_images[index] = np.array(image_data) / 255.0
    return x_train_images, x_dev_images, x_test_images
