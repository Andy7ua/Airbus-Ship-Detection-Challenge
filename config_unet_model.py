import random
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from preprocessing import rle_to_mask

# Train Unet Model
random.seed(33)
TRAIN_DIR = '/Users/andy/Downloads/airbus-ship-detection/train_v2/'
TEST_DIR = '/Users/andy/Downloads/airbus-ship-detection/test_v2/'


df = pd.read_csv("train_ship_segmentations_v2.csv")
df['EncodedPixels'] = df['EncodedPixels'].astype('string')

# Delete corrupted images
CORRUPTED_IMAGES = ['6384c3e78.jpg']
df = df.drop(df[df['ImageId'].isin(CORRUPTED_IMAGES)].index)

# Dataframe that contains the segmentation for each ship in the image.
instance_segmentation = df

# Dataframe that contains the segmentation of all ships in the image.
image_segmentation = df.groupby(by=['ImageId'])['EncodedPixels'].apply(lambda x: np.nan if pd.isna(x).any() else ' '.join(x)).reset_index()


# Data preprocessing
IMAGES_WITHOUT_SHIPS_NUMBER = 25000

# reduce the number of images without ships
images_without_ships = image_segmentation[image_segmentation['EncodedPixels'].isna()]['ImageId'].values[:IMAGES_WITHOUT_SHIPS_NUMBER]
images_with_ships = image_segmentation[image_segmentation['EncodedPixels'].notna()]['ImageId'].values
images_list = np.append(images_without_ships, images_with_ships)

# remove corrupted images
images_list = np.array(list(filter(lambda x: x not in CORRUPTED_IMAGES, images_list)))

VALIDATION_LENGTH = 2000
TEST_LENGTH = 2000
TRAIN_LENGTH = len(images_list) - VALIDATION_LENGTH - TEST_LENGTH
BATCH_SIZE = 16
BUFFER_SIZE = 1000
IMG_SHAPE = (256, 256)
NUM_CLASSES = 2


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a])


def load_train_image(tensor) -> tuple:
    path = tf.get_static_value(tensor).decode("utf-8")

    image_id = path.split('/')[-1]
    input_image = cv2.imread(path)
    input_image = tf.image.resize(input_image, IMG_SHAPE)
    input_image = tf.cast(input_image, tf.float32) / 255.0

    encoded_mask = image_segmentation[image_segmentation['ImageId'] == image_id].iloc[0]['EncodedPixels']
    input_mask = np.zeros(IMG_SHAPE + (1,), dtype=np.int8)
    if not pd.isna(encoded_mask):
        input_mask = rle_to_mask(encoded_mask)
        input_mask = cv2.resize(input_mask, IMG_SHAPE, interpolation=cv2.INTER_AREA)
        input_mask = np.expand_dims(input_mask, axis=2)
    one_hot_segmentation_mask = one_hot(input_mask, NUM_CLASSES)
    input_mask_tensor = tf.convert_to_tensor(one_hot_segmentation_mask, dtype=tf.float32)

    class_weights = tf.constant([0.0005, 0.9995], tf.float32)
    sample_weights = tf.gather(class_weights, indices=tf.cast(input_mask_tensor, tf.int32), name='cast_sample_weights')

    return input_image, input_mask_tensor, sample_weights


# Creating TensorFlow datasets
images_list = tf.data.Dataset.list_files([f'{TRAIN_DIR}{name}' for name in images_list], shuffle=True)
train_images = images_list.map(lambda x: tf.py_function(load_train_image, [x], [tf.float32, tf.float32]), num_parallel_calls=tf.data.AUTOTUNE)

validation_dataset = train_images.take(VALIDATION_LENGTH)
test_dataset = train_images.skip(VALIDATION_LENGTH).take(TEST_LENGTH)
train_dataset = train_images.skip(VALIDATION_LENGTH + TEST_LENGTH)

train_batches = (
    train_dataset
    .repeat()
    .batch(BATCH_SIZE))

validation_batches = validation_dataset.batch(BATCH_SIZE)

test_batches = test_dataset.batch(BATCH_SIZE)
