"""
DogFaceNet
Landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import os
import numpy as np
import tensorflow as tf

import models


PATH = '../data/landmarks/'
SPLIT = 0.8
BATCH_SIZE = 16
EPOCHS = 100
STEPS_PER_EPOCH = 40


############################################################
#  Data pre-processing
############################################################

resized_path = PATH + 'resized/'
filenames = os.listdir(resized_path)

filename_int = np.sort([int(s[:-4]) for s in filenames])

filenames = np.array([resized_path + str(i) + '.jpg' for i in filename_int])

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


# base_model = tf.keras.applications.ResNet50(include_top=False, pooling='avg')
# x = tf.keras.layers.Dense(1024, activation='relu')(base_model.output)
# x = tf.keras.layers.Dropout(0.25)(x)
# out = tf.keras.layers.Dense(14, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)

# model = tf.keras.Model(inputs=base_model.input, outputs=out)
#for layer in base_model.layers: layer.trainable = False


layers = [10, 20, 40, 80, 160]
model = models.ResNet(layers, 14, (100,100,3,))

print(model.summary())


model.compile(optimizer=tf.keras.optimizers.Adam(0.01, decay=1e-5),
              loss='mse',       # mean squared error
              metrics=['mae']) 



############################################################
#  Training
############################################################

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        '../weights/landmarks/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
        save_best_only=True,
        save_weights_only=True
        ),
    tf.keras.callbacks.TensorBoard(log_dir='../output/logs')
    ]

model.fit(
    data_train,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=data_valid,
    validation_steps=3,
    callbacks=callbacks
    )

#model.save_weights('../weights/landmarks/')


