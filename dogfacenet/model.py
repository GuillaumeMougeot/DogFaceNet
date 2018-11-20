"""
DogFaceNet
The main DogFaceNet implementation

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import os
import numpy as np
import matplotlib.pyplot as plt

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

filename_dog1 = []
for file in os.listdir(PATH_DOG1):
    if ".jpg" in file:
        filenames_bg += [file]


# Load images into Numpy arrays

images_bg = KI.img_to_array(KI.load_img(PATH_BG + filenames_bg[0]))


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
    #model = tf.keras.Model(inputs=base_model.input, outputs=x)
    return x


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