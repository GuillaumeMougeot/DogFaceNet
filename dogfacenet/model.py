"""
DogFaceNet
The main DogFaceNet implementation

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import tensorflow as tf

import keras.models as KM
import keras.applications as KA
import keras.layers as KL


# Paths of images folders
PATH_BG = "..\\data\\bg\\"
PATH_DOG1 = "..\\data\\dog1\\"

# Images parameters for network feeding
IM_H = 224
IM_W = 224
IM_C = 3

# Training parameters:
EPOCHS = 1
BATCH_SIZE = 32

# Embedding size
EMB_SIZE = 128


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

# Opens an image file, stores it into a tf.Tensor and reshapes it
def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [IM_H, IM_W])
        return image_resized, label

filenames = np.append(
        [PATH_DOG1 + filenames_dog1[i] for i in range(len(filenames_dog1))],
        [PATH_BG + filenames_bg[i] for i in range(len(filenames_bg))],
        axis=0
        )
labels = np.append(np.ones(len(filenames_dog1)), np.arange(2,2+len(filenames_bg)))

# Filenames and labels place holder
filenames_placeholder = tf.placeholder(filenames.dtype, filenames.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

# Defining dataset
dataset = tf.data.Dataset.from_tensor_slices((filenames_placeholder, (filenames_placeholder, labels_placeholder)))
dataset = dataset.map(_parse_function)

# Batch the dataset for training
data_train = dataset.repeat(EPOCHS).batch(BATCH_SIZE)
iterator = data_train.make_initializable_iterator()
next_element = iterator.get_next()

# Define the dataset for loss computation
# embedding_placeholder = tf.placeholder(tf.float32, shape=(None, EMB_SIZE))
# data_loss = tf.data.Dataset.from_tensor_slices((filenames_placeholder, (labels_placeholder, embedding_placeholder)))

# Defining the global prediction dictionary
data_dict = np.concatenate(
        (np.expand_dims(filenames,1), np.expand_dims(labels,1)),
        axis = 1
        )

dict_pred = {filename:(label, np.random.random_sample((EMB_SIZE,))) for filename, label in data_dict}


############################################################
#  NASNet Graph
############################################################


# Build the model using Keras pretrained model NASNetMobile,
# a light and efficient network
# def NASNet_embedding(
#         input_tensor,
#         input_shape=(224,224,3),
#         include_top=False,
#         training=True
#         ):

#         base_model = KA.NASNetMobile(
#                 input_tensor=input_tensor,
#                 input_shape=input_shape,
#                 include_top=False
#                 )
#         x = KL.GlobalAveragePooling2D()(base_model.output)
#         x = KL.Dense(1056, activation='relu')(x)
#         if training:
#                 x = KL.Dropout(0.5)(x)
#         x = KL.Dense(EMB_SIZE)(x)
#         x = tf.keras.backend.l2_normalize(x)

#         return x

class NASNet_embedding(tf.keras.Model):
        def __init__(self):
                self.pool = KL.GlobalAveragePooling2D()
                self.dense_1 = KL.Dense(1056, activation='relu')
                self.dropout = KL.Dropout(0.5)
                self.dense_2 = KL.Dense(EMB_SIZE)
        
        def __call__(self, input_tensor, input_shape=(224,224,3), training=False):
                base_model = KA.NASNetMobile(
                        input_tensor=input_tensor,
                        input_shape=input_shape,
                        include_top=False
                        )
                x = self.pool(base_model.output)
                x = self.dense_1(x)
                if training:
                        x = self.dropout(x)
                x = self.dense_2(x)

                return tf.keras.backend.l2_normalize(x)


# Predict
y_pred = NASNet_embedding(next_element[0])
filenames_pred, labels_pred = next_element[1]
sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(iterator.initializer, feed_dict={filenames_placeholder:filenames, labels_placeholder:labels})
sess.run(init)
print(sess.run(y_pred))
print(filenames_pred, labels_pred)

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


def deviation_loss(dict_pred):
        sum_class_loss = 0
        classes_loss = 0

        class_pred = {}

        # Compute all center of mass
        for _, (label, pred) in dict_pred:
                if label in class_pred.keys():
                        class_pred[label][0] += pred
                        class_pred[label][1] += 1
                else:
                        class_pred[label] = (pred,1)
        for label in class_pred:
                class_pred[label][0] /= class_pred[label][1]

        # Compute all classes center of mass
        class_pred_values = np.array(class_pred.values())
        classes_center = np.sum(class_pred_values)/len(class_pred)
        classes_loss -= np.sum(np.log(np.linalg.norm(class_pred_values - classes_center)))
        
        # Compute 
        for _, (label, pred) in dict_pred:
                sum_class_loss += np.linalg.norm(pred - class_pred[label])

        return classes_loss + sum_class_loss