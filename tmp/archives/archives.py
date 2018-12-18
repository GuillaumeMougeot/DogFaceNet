from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from tqdm import tqdm

import tensorflow as tf

import keras.models as KM
import keras.applications as KA
import keras.layers as KL
import keras.preprocessing.image as KI


PATH_BG = "..\\data\\bg\\"
PATH_DOG1 = "..\\data\\dog1\\"


############################################################
#  Data analysis: Archive
############################################################

# Retrieve filenames
filenames_bg = []
for file in os.listdir(PATH_BG):
    if ".jpg" in file:
        filenames_bg += [file]

filenames_dog1 = []
for file in os.listdir(PATH_DOG1):
    if ".jpg" in file:
        filenames_dog1 += [file]


IM_H = 224
IM_W = 224
IM_C = 3


#### Method 2: using standard techniques

# Load images into Numpy arrays

def load_imgs(path, filenames, new_shape=(224,224), mode='constant'):
    """Load an image defined by filenames from a certain path and reshape it 
    with the precised mode"""

    image = KI.img_to_array(KI.load_img(path + filenames[0]))/255
    image = resize(image, new_shape, mode=mode)
    images = np.array([image])
    for i in tqdm(range(1, len(filenames))):
        image = KI.img_to_array(KI.load_img(path + filenames[i]))/255
        image = resize(image, new_shape, mode=mode)
        images = np.append(images, [image], axis=0)
    return images

# Load background images
images_bg = load_imgs(PATH_BG, filenames_bg, (IM_H, IM_W))

# Load dog 1 images
images_dog1 = load_imgs(PATH_DOG1, filenames_dog1, (IM_H, IM_W))

# Define the identies of the dogs: dog1->1, bg->[2,len(bg)]
labels_bg = np.arange(2, len(images_bg)+2)
labels_dog1 = np.ones(len(images_dog1))

# Queue images and labels
images = np.append(images_dog1, images_bg)
labels = np.append(labels_dog1, labels_bg)

# Method 1: Load data into Tensorflow tensor
# dataset = tf.data.Dataset.from_tensor_slices((images, labels))

# Method 2: Use placeholder
assert images.shape[0] == labels.shape[0]

# Graph elements:
images_placeholder = tf.placeholder(images.dtype, images.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

# Graph elements:
dataset = tf.data.Dataset.from_tensor_slices((images_placeholder, labels_placeholder))

# Graph elements:
iterator = dataset.make_initializable_iterator()

# sess.run(iterator.initializer, feed_dict={images_placeholder: images, labels_placeholder: labels})

#### End Method 2