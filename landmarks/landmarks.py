"""
DogFaceNet
Landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import os
import numpy as np
import tensorflow as tf

from models import ConvNet


PATH = '../data/landmarks/'
SPLIT = 0.8
BATCH_SIZE = 32
EPOCHS = 100


############################################################
#  Data pre-processing
############################################################


filenames = os.listdir(PATH + 'resized/')
filenames = np.array([PATH + 'resized/' + file for file in filenames])

labels = np.load(PATH + 'resized_labels.npy')


assert len(filenames)==len(labels)

train_split = int(SPLIT*len(filenames))

filenames_train = filenames[:train_split]
filenames_valid = filenames[train_split:]

labels_train = labels[:train_split]
labels_valid = labels[train_split:]

def _parse(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.cast(image_decoded, tf.float32)
    label = tf.reshape(label, [14])
    label = tf.cast(label, tf.float32)
    return image_decoded, label

data_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
data_train = data_train.map(_parse).batch(BATCH_SIZE).repeat()

data_valid = tf.data.Dataset.from_tensor_slices((filenames_valid, labels_valid))
data_valid = data_valid.map(_parse).batch(BATCH_SIZE).repeat()


############################################################
#  Model definition
############################################################

layers = [20, 30, 40, 80, 120, 200]

model = ConvNet(layers, 14, (500,500,3,))

print(model.summary())

model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae']) 



############################################################
#  Training
############################################################

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '../weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=True,
        save_weights_only=True
        ),
    tf.keras.callbacks.TensorBoard(log_dir='../output/logs')
    ]

model.fit(
    data_train,
    epochs=EPOCHS,
    steps_per_epoch=30,
    validation_data=data_valid,
    validation_steps=3,
    callbacks=callbacks
    )

#model.save_weights('../weights/landmarks/')


