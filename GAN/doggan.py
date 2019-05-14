"""
DogGAN
A GAN for dog face generation.

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
# from triplets_processing import *
# from online_training import *
# from triplet_loss import *

PATH = '../data/dogfacenet/aligned/after_4_bis/'
PATH_SAVE = '../output/history/'
PATH_MODEL = '../output/model/gan/'
SIZE = (224,224,3)
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1
print(len(labels))

nbof_classes = len(np.unique(labels))
print(nbof_classes)

nbof_test = int(TEST_SPLIT*nbof_classes)

keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

alpha = 0.3
def triplet(y_true,y_pred):
    
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)
    
    return K.less(ap+alpha,an)

"""
Models definition
"""


from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    AveragePooling2D,
    Add,
    Concatenate,
    GlobalAveragePooling2D,
    DepthwiseConv2D,
    LeakyReLU,
    UpSampling2D
    )
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Lambda, BatchNormalization, Reshape


layers = [16,32,64,128,256,256]

### Discriminator

inputs = Input(shape=SIZE)

x = Conv2D(16, (1, 1), padding='same')(inputs)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)

x = AveragePooling2D((2,2))(x)

for i in range(len(layers)-1):

    x = Conv2D(layers[i], (3,3), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = Conv2D(layers[i+1], (3,3), padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = BatchNormalization()(x)

    x = AveragePooling2D((2,2))(x)
    
x = Conv2D(256, (3,3), padding='valid')(x)
x = LeakyReLU(0.2)(x)
x = BatchNormalization()(x)
x = Flatten()(x)
x = Dropout(0.5)(x)
outputs = Dense(1, use_bias=True, activation='sigmoid')(x)

dis = tf.keras.Model(inputs,outputs)

dis.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001,beta_1=0,beta_2=0.99,epsilon=1e-8),
              metrics=['acc'])

print("Discriminator model:")
print(dis.summary())

### Generator

inputs = Input(shape=(1,1,256))

x = BatchNormalization()(inputs)
x = LeakyReLU(0.2)(x)
x = Conv2DTranspose(256, (3,3), padding='valid')(x)
# x = LeakyReLU(0.2)(x)
# x = BatchNormalization()(x)

for i in range(len(layers)-2,-1,-1):
    x = UpSampling2D((2,2))(x)

    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(layers[i], (3,3), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(layers[i], (3,3), padding='same')(x)
    # x = LeakyReLU(0.2)(x)
    # x = BatchNormalization()(x)

x = UpSampling2D((2,2))(x)

x = BatchNormalization()(x)
x = LeakyReLU(0.2)(x)
outputs = Conv2D(3, (1, 1), padding='same')(x)

gen = tf.keras.Model(inputs,outputs)

gen.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(0.001,beta_1=0,beta_2=0.99,epsilon=1e-8),
              metrics=['acc'])

print("Generator model:")
print(gen.summary())



