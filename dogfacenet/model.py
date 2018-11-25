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
dataset = tf.data.Dataset.from_tensor_slices((filenames_placeholder, labels_placeholder))
dataset = dataset.map(_parse_function)

# Batch the dataset for training
data_train = dataset.repeat(EPOCHS).batch(BATCH_SIZE)
iterator = data_train.make_initializable_iterator()
next_element = iterator.get_next()

# Define the global step and dropout rate
global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)


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
                super(NASNet_embedding, self).__init__(name='')

                self.pool = KL.GlobalAveragePooling2D()
                self.dense_1 = KL.Dense(1056, activation='relu')
                self.dropout = KL.Dropout(0.5)
                self.dense_2 = KL.Dense(EMB_SIZE)
        
        def __call__(self, input_tensor, input_shape=(224,224,3), training=False):
                # base_model = KA.NASNetMobile(
                #         input_tensor=input_tensor,
                #         input_shape=input_shape,
                #         include_top=False
                #         )
                # x = self.pool(base_model.output)
                x = self.pool(input_tensor)
                x = self.dense_1(x)
                if training:
                        x = self.dropout(x)
                x = self.dense_2(x)

                return tf.keras.backend.l2_normalize(x)


# Predict
model = NASNet_embedding()


############################################################
#  Training session
############################################################


init = tf.global_variables_initializer()

with tf.Session() as sess:
        sess.run(iterator.initializer, feed_dict={filenames_placeholder:filenames, labels_placeholder:labels})
        sess.run(init)
        print(sess.run(y_pred))


############################################################
#  Loss Functions
############################################################

def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('arcface_loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keep_dims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output


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