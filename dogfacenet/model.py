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
PATH_BG = "..\\data\\bg\\"
PATH_DOG1 = "..\\data\\dog1\\"

# Images parameters for network feeding
IM_H = 224
IM_W = 224
IM_C = 3

# Training parameters:
EPOCHS = 10
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
labels_placeholder = tf.placeholder(tf.int64, labels.shape)

# Defining dataset
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


class NASNet_embedding(tf.keras.Model):
        def __init__(self):
                super(NASNet_embedding, self).__init__(name='')

                self.pool = tf.keras.layers.GlobalAveragePooling2D()
                self.dense_1 = tf.layers.Dense(1056, activation='relu')
                self.dropout = tf.layers.Dropout(0.5)
                self.dense_2 = tf.layers.Dense(EMB_SIZE)
        
        def __call__(self, input_tensor, input_shape=(224,224,3), training=False, unfreeze=True):
                # base_model = KA.NASNetMobile(
                #         input_tensor=input_tensor,
                #         input_shape=input_shape,
                #         include_top=False
                #         )
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

logit = arcface_loss(embedding=output, labels=next_labels, w_init=None, out_num=len(labels))
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

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

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
                feed_dict = {filenames_placeholder:filenames, labels_placeholder:labels}
                sess.run(iterator.initializer, feed_dict=feed_dict)
                while True:
                        try:
                                _, loss_value, summary_op_value = sess.run((train, loss, summary_op))
                                summary.add_summary(summary_op_value, count)
                                count += 1
                                print(loss_value)
                        except tf.errors.OutOfRangeError:
                                break
