"""
DogFaceNet
The main DogFaceNet implementation

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

import tensorflow as tf

import keras.applications as KA
import keras.layers as KL
import keras.preprocessing.image as KI


PATH_BG = "..\\data\\bg\\"
PATH_DOG1 = "..\\data\\dog1\\"


############################################################
#  Data analysis
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

# image = tf.image.decode_jpeg(PATH_BG + filenames_bg[0], channels=3)
# resized_image = tf.image.resize_images(image, [IM_H, IM_W])
# print(resized_image)

# Load images into Numpy arrays

# Load background images
image = KI.img_to_array(KI.load_img(PATH_BG + filenames_bg[0]))/255
image = resize(image, (IM_H, IM_W), mode='reflect')
images_bg = np.array([image])
for i in range(1, len(filenames_bg)):
    image = KI.img_to_array(KI.load_img(PATH_BG + filenames_bg[i]))/255
    image = resize(image, (IM_H, IM_W), mode='reflect')
    np.append(images_bg, [image], axis=0)

# Load dog 1 images
image = KI.img_to_array(KI.load_img(PATH_DOG1 + filenames_dog1[0]))/255
image = resize(image, (IM_H, IM_W), mode='reflect')
images_dog1 = np.array([image])
for i in range(1, len(filenames_dog1)):
    image = KI.img_to_array(KI.load_img(PATH_DOG1 + filenames_dog1[i]))/255
    image = resize(image, (IM_H, IM_W), mode='reflect')
    np.append(images_dog1, [image], axis=0)


############################################################
#  NASNet Graph
############################################################

# Build the model
def NASNet_embedding(
  input_shape=(224,224,3),
  include_top=False
):
    base_model = KA.NASNetMobile(input_shape=(224,224,3), include_top=False)

    x = base_model.output
    x = KL.GlobalAveragePooling2D()(x)
    x = KL.Dense(1056, activation='relu')(x)
    x = KL.Dropout(0.5)(x)
    x = KL.Dense(128)(x)

    return x

# With earger execution:
# tf.enable_eager_execution()

# model = tf.keras.Sequential([
#   KA.NASNetMobile(input_shape=(224,224,3), include_top=False),
#   KL.GlobalAveragePooling2D(),
#   KL.Dense(1056, activation='relu'),
#   KL.Dense(128)
# ])


############################################################
#  Loss Functions
############################################################

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss