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

from losses import arcface
from dataset import get_dataset

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


tf.enable_eager_execution()

############################################################
#  Data pre-processing
############################################################


# Retrieve dataset from folders
# filenames_train, labels_train, filenames_valid, labels_valid = get_dataset(
#     PATH_BG, PATH_DOG1, TRAIN_SPLIT)
filenames_train, labels_train, filenames_valid, labels_valid, count_labels = get_dataset()

# Defining dataset

# Opens an image file, stores it into a tf.Tensor and reshapes it
def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [IM_H, IM_W])
    return image_resized, label

def _parse_fn(filename):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_resized = tf.image.resize_images(image_decoded, [IM_H, IM_W])
    return image_resized


data_train = tf.data.Dataset.from_tensor_slices(
    (tf.constant(filenames_train),
    tf.constant(labels_train))
    )
data_train = data_train.map(_parse_function)
data_train = data_train.shuffle(1000).batch(BATCH_SIZE)

x_train = tf.constant(filenames_train)
images_train = tf.map_fn(_parse_fn, x_train, dtype=tf.float32)
y_train = tf.constant(labels_train)

data_valid = tf.data.Dataset.from_tensor_slices(
    (tf.constant(filenames_valid),
    tf.constant(labels_valid))
    )
data_valid = data_valid.map(_parse_function)

x_valid = tf.constant(filenames_valid)
images_valid = tf.map_fn(_parse_fn, x_valid, dtype=tf.float32)
y_valid = tf.constant(labels_valid)
print(images_train.shape)
############################################################
#  Models
############################################################


class Dummy_embedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(Dummy_embedding, self).__init__(name='dummy')
        self.conv1 = tf.keras.layers.Conv2D(10,(3, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(20,(3, 3))
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(40,(3, 3))
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(80,(3, 3))
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.layers.Dense(emb_size)
    
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.avg_pool(x)
        x = self.dense(x)

        return tf.nn.l2_normalize(x)

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

class ResnetConvBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv1(input_tensor)
        shortcut = self.bn1(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)

class ResNet_embedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(ResNet_embedding, self).__init__(name='resnet')
        self.conv1_pad = tf.keras.layers.ZeroPadding2D(padding=(3,3))
        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))
        self.bn_conv1 = tf.keras.layers.BatchNormalization()

        self.pool1_pad = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.pool1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        #filters = [[64,64,256], [128,128,512], [256,256,1024], [512,512,2048]]
        #nrof_identity_block = [2,3,5,2]
        filters = [[64,64,256]]
        nrof_identity_block = [1]

        self.in_layers = []
        for i in range(len(filters)):
            self.in_layers += [ResnetConvBlock(3, filters[i])]
            for _ in range(nrof_identity_block[i]):
                self.in_layers += [ResnetIdentityBlock(3, filters[i])]
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.embedding = tf.keras.layers.Dense(emb_size)

    def __call__(self, input_tensor=None, training=False):
        x = self.conv1_pad(input_tensor)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1_pad(x)
        x = self.pool1(x)

        for in_layer in self.in_layers:
            x = in_layer(x, training=training)
        
        x = self.avg_pool(x)
        x = self.embedding(x)

        return tf.nn.l2_normalize(x)


class NASNet_embedding(tf.keras.Model):
    def __init__(self):
        super(NASNet_embedding, self).__init__(name='')

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_1 = tf.layers.Dense(1056, activation='relu')
        self.dropout = tf.layers.Dropout(0.5)
        self.dense_2 = tf.layers.Dense(EMB_SIZE)

    def __call__(self, input_tensor, input_shape=(224, 224, 3), training=True, unfreeze=True):
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


model = Dummy_embedding(EMB_SIZE)

model.compile(
    optimizer=tf.train.AdamOptimizer(), 
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks=[
    tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='../output/summary')
]

steps_per_epoch=len(filenames_train)//BATCH_SIZE + 1

model.fit(
    data_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks,
    validation_data=(images_valid, y_valid)
)
