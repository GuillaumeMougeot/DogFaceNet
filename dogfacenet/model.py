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

from losses import arcface_loss


# Paths of images folders
PATH_BG = "../data/bg/"
PATH_DOG1 = "../data/dog1/"

# Images parameters for network feeding
IM_H = 224
IM_W = 224
IM_C = 3

# Training parameters:
EPOCHS = 10
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8

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

# Splits the dataset

split_dog1 = TRAIN_SPLIT * len(filenames_dog1)
split_bg = TRAIN_SPLIT * len(filenames_bg) 

## Training set

filenames_dog1_train = filenames_dog1[:split_dog1]
filenames_bg_train = filenames_bg[:split_bg]

filenames_train = np.append(
        [PATH_DOG1 + filenames_dog1_train[i] for i in range(len(filenames_dog1_train))],
        [PATH_BG + filenames_bg_train[i] for i in range(len(filenames_bg_train))],
        axis=0
        )
# labels_train = [1,1,1,...,1,2,3,...,len(filenames_bg_train)] <-- there are len(filenames_dog1_train) ones
labels_train = np.append(np.ones(len(filenames_dog1_train)), np.arange(2,2+len(filenames_bg_train)))

assert len(filenames_train)==len(labels_train)

## Validation set
filenames_dog1_valid = filenames_dog1[split_dog1:]
filenames_bg_valid = filenames_bg[split_bg:]

filenames_valid = np.append(
        [PATH_DOG1 + filenames_dog1_valid[i] for i in range(len(filenames_dog1_valid))],
        [PATH_BG + filenames_bg_valid[i] for i in range(len(filenames_bg_valid))],
        axis=0
        )

# labels_valid = [1,1,1,...,1,labels_train[-1]+1,labels_train[-1]+2,...,len(filenames_bg_valid)+labels_train[-1]+1]
# <-- there are len(filenames_dog1_valid) ones
labels_valid = np.append(np.ones(len(filenames_dog1_valid)), np.arange(labels_train[-1]+1,labels_train[-1]+1+len(filenames_bg_valid)))



# Filenames and labels place holder
filenames_placeholder = tf.placeholder(filenames_train.dtype, filenames_train.shape)
labels_placeholder = tf.placeholder(tf.int64, labels_train.shape)

# Defining dataset

# Opens an image file, stores it into a tf.Tensor and reshapes it
def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [IM_H, IM_W])
        return image_resized, label

dataset = tf.data.Dataset.from_tensor_slices((filenames_placeholder, labels_placeholder))
dataset = dataset.map(_parse_function)

# Batch the dataset for training
data_train = dataset.batch(BATCH_SIZE)
iterator = data_train.make_initializable_iterator()
next_element = iterator.get_next()

# Define the global step and dropout rate
global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
dropout_rate = tf.placeholder(name='dropout_rate', dtype=tf.float32)


############################################################
#  NASNet Graph
############################################################


class NASNet_embedding(tf.keras.models.Model):
        def __init__(self):
                super(NASNet_embedding, self).__init__(name='')

                self.pool = tf.keras.layers.GlobalAveragePooling2D()
                self.dense_1 = tf.layers.Dense(1056, activation='relu')
                self.dropout = tf.layers.Dropout(0.5)
                self.dense_2 = tf.layers.Dense(EMB_SIZE)
        
        def __call__(self, input_tensor, input_shape=(224,224,3), training=False, unfreeze=True):
                # base_model = tf.keras.applications.NASNetMobile(
                #         input_tensor=input_tensor,
                #         input_shape=input_shape,
                #         include_top=False
                #         )

                # for layer in base_model.layers: layer.trainable = False
                # x = self.pool(base_model.output)
                x = self.pool(input_tensor)
                x = self.dense_1(x)
                if training:
                        x = self.dropout(x)
                x = self.dense_2(x)

                return tf.keras.backend.l2_normalize(x)

model = NASNet_embedding()

next_images, next_labels = next_element

output = model(next_images)

logit = arcface_loss(embedding=output, labels=next_labels, w_init=None, out_num=len(labels_train))
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=next_labels))

# Optimizer
lr = 0.01

opt = tf.train.AdamOptimizer(learning_rate=lr)
train = opt.minimize(loss)

# Accuracy for validation and testing
pred = tf.nn.softmax(logit)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), next_labels), dtype=tf.float32))


############################################################
#  Training session
############################################################


init = tf.global_variables_initializer()

with tf.Session() as sess:

        summary = tf.summary.FileWriter('../output/summary', sess.graph)
        summaries = []
        for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))
        summaries.append(tf.summary.scalar('inference_loss', loss))
        summary_op = tf.summary.merge(summaries)
        saver = tf.train.Saver(max_to_keep=100)

        sess.run(init)    

        count = 0

        for i in range(EPOCHS):
                feed_dict = {filenames_placeholder:filenames_train, labels_placeholder:labels_train}
                sess.run(iterator.initializer, feed_dict=feed_dict)
                while True:
                        try:
                                _, loss_value, summary_op_value = sess.run((train, loss, summary_op))
                                summary.add_summary(summary_op_value, count)
                                count += 1
                                print(loss_value)
                        except tf.errors.OutOfRangeError:
                                break
