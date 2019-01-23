"""
DogFaceNet
The main DogFaceNet implementation

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import tensorflow as tf

from losses import arcface_loss
from dataset import get_dataset
from models import Dummy_embedding

# Paths of images folders
PATH_BG = "../data/bg/"
PATH_DOG1 = "../data/dog1/"

# Images parameters for network feeding
IM_H = 224
IM_W = 224
IM_C = 3

# Training parameters:
EPOCHS = 100
BATCH_SIZE = 32
TRAIN_SPLIT = 0.8

# Embedding size
EMB_SIZE = 128


############################################################
#  Data pre-processing
############################################################


# Retrieve dataset from folders
filenames_train, labels_train, filenames_valid, labels_valid, count_labels = get_dataset()

# Filenames and labels placeholders
filenames_train_placeholder = tf.placeholder(filenames_train.dtype, filenames_train.shape)
labels_train_placeholder = tf.placeholder(tf.int64, labels_train.shape)

filenames_valid_placeholder = tf.placeholder(filenames_valid.dtype, filenames_valid.shape)
labels_valid_placeholder = tf.placeholder(tf.int64, labels_valid.shape)

# Opens an image file, stores it into a tf.Tensor and reshapes it
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [IM_H, IM_W])
    return image_resized, label

# Datasets initializing
data_train = tf.data.Dataset.from_tensor_slices((filenames_train_placeholder, labels_train_placeholder))
data_train = data_train.map(_parse_function)

data_valid = tf.data.Dataset.from_tensor_slices((filenames_valid_placeholder,labels_valid_placeholder))
data_valid = data_valid.map(_parse_function)

# Batch the datasets for training and validation
data_train = data_train.shuffle(1000).batch(BATCH_SIZE)

data_valid = data_valid.shuffle(1000).batch(BATCH_SIZE)

# Reinitializable iterator
iterator = tf.data.Iterator.from_structure(data_train.output_types, data_train.output_shapes)
next_element = iterator.get_next()

train_init_op = iterator.make_initializer(data_train)
valid_init_op = iterator.make_initializer(data_valid)


############################################################
#  Graph of the model
############################################################


model = Dummy_embedding(EMB_SIZE)

# Training
next_images, next_labels = next_element

output = model(next_images)

logit = arcface_loss(embedding=output, labels=next_labels,
                     w_init=None, out_num=count_labels)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logit, labels=next_labels))

# Optimizer
lr = 0.01

opt = tf.train.AdamOptimizer(learning_rate=lr)
train = opt.minimize(loss)

# Accuracy for validation and testing
pred = tf.nn.softmax(logit)
acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmin(pred, axis=1), next_labels), dtype=tf.float32))


############################################################
#  Training session
############################################################


init = tf.global_variables_initializer()

with tf.Session() as sess:

    # summary = tf.summary.FileWriter('../output/summary', sess.graph)
    # summaries = []
    # for var in tf.trainable_variables():
    #     summaries.append(tf.summary.histogram(var.op.name, var))
    # summaries.append(tf.summary.scalar('inference_loss', loss))
    # summary_op = tf.summary.merge(summaries)
    # saver = tf.train.Saver(max_to_keep=100)

    sess.run(init)

    # Training
    nrof_batches = len(filenames_train)//BATCH_SIZE + 1
    nrof_batches_valid = len(filenames_valid)//BATCH_SIZE + 1

    print("Start of training...")
    for i in range(EPOCHS):

        # Training
        feed_dict_train = {filenames_train_placeholder: filenames_train,
                     labels_train_placeholder: labels_train}

        sess.run(train_init_op, feed_dict=feed_dict_train)

        for j in trange(nrof_batches):
            try:
                _, loss_value, acc_value = sess.run((train, loss, acc))
                # summary.add_summary(summary_op_value, count)
                tqdm.write("\n Batch: " + str(j)
                    + ", Loss: " + str(loss_value)
                    + ", Accuracy: " + str(acc_value)
                    )

            except tf.errors.OutOfRangeError:
                break
        
        # Validation
        feed_dict_valid = {filenames_valid_placeholder: filenames_valid,
                           labels_valid_placeholder: labels_valid}

        sess.run(valid_init_op, feed_dict=feed_dict_valid)
        print("Start validation...")
        tot_acc = 0
        for _ in trange(nrof_batches_valid):
            try:
                loss_valid_value, acc_valid_value = sess.run((loss, acc))
                tot_acc += acc_valid_value
                tqdm.write("Loss: " + str(loss_valid_value)
                    + ", Accuracy: " + str(acc_valid_value)
                    )

            except tf.errors.OutOfRangeError:
                break
        print("End of validation. Total accuray: " + str(tot_acc/nrof_batches_valid))


    print("End of training.")
    print("Start evaluation...")
    # Evaluation on the validation set:
    ## One-shot training
    #sess.run()


