"""
DogFaceNet
Landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import os
import numpy as np
import tensorflow as tf
import skimage as sk
from tqdm import tqdm

import models

PATH = '../data/landmarks/'

IMAGE_SIZE = (128,128,3)

# Limite size of input for testing
TEST_LIMIT = 1000

SPLIT = 0.8
BATCH_SIZE = 32
EPOCHS = 20
STEPS_PER_EPOCH = 40


############################################################
#  Data pre-processing
############################################################

# Retrieve the raw images
resized_path = PATH + 'resized/'
filenames = os.listdir(resized_path)

## Sort the filenames in the proper order
filename_int = np.sort([int(s[:-4]) for s in filenames])

filenames = np.array([resized_path + str(i) + '.jpg' for i in filename_int])

# Retrieve the masks
masks_path = PATH + 'resized_masks/'
f_masks = os.listdir(masks_path)

## Sort the filenames in the proper order
f_masks_int = np.sort([int(s[:-4]) for s in f_masks])

f_masks = np.array([resized_path + str(i) + '.jpg' for i in f_masks_int])

# Retrieve the landmarks
labels = np.load(PATH + 'resized_labels.npy')

# Reshape the outputs
print(labels.shape)
print(f_masks.shape)
print(filenames.shape)
assert len(f_masks)==len(labels)
assert len(filenames)==len(labels)

if TEST_LIMIT>len(labels):
    TEST_LIMIT = len(labels)


w,h,c = IMAGE_SIZE
images = np.empty((0,w,h,c))

masks = np.empty((0,w//4,h//4,1))

print("Loading images...")
labels = labels[:TEST_LIMIT]
labels = np.reshape(labels,(TEST_LIMIT,14))
labels = labels[:,:10]

for i in tqdm(range(TEST_LIMIT)):
    image = sk.io.imread(filenames[i])
    image_resized = sk.transform.resize(image, IMAGE_SIZE)
    images = np.vstack((images,np.expand_dims(image_resized,0)))

    mask = sk.io.imread(f_masks[i])

    mask_resized = sk.transform.resize(mask, (w//4,h//4,1))
    masks = np.vstack((masks,np.expand_dims(mask_resized,0)))



train_split = int(SPLIT*len(labels))

images_train = images[:train_split]
images_valid = images[train_split:]

masks_train = masks[:train_split]
masks_valid = masks[train_split:]

labels_train = labels[:train_split]
labels_valid = labels[train_split:]

# filenames_train = filenames[:train_split]
# filenames_valid = filenames[train_split:]

# f_masks_train = f_masks[:train_split]
# f_masks_valid = f_masks[train_split:]

# labels_train = labels[:train_split]
# labels_valid = labels[train_split:]

# def _parse(filename, label):

#     # Decode image
#     image_string = tf.read_file(filename)
#     image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#     image_decoded = tf.cast(image_decoded, tf.float32)
#     image_resized = tf.image.resize_bilinear(image_decoded, (227,227,3))

#     # Decode mask
#     mask_string = tf.read_file(mask)
#     mask_decoded = tf.image.decode_jpeg(mask_string, channels=1)
#     mask_decoded = tf.cast(mask_decoded, tf.float32)

#     # Decode label
#     label = tf.reshape(label, [14])
#     label = tf.cast(label, tf.float32)
#     return image_decoded, label

# data_train = tf.data.Dataset.from_tensor_slices((filenames_train, labels_train))
# data_train = data_train.map(_parse).batch(BATCH_SIZE).repeat()

# data_valid = tf.data.Dataset.from_tensor_slices((filenames_valid, masks_labels_valid))
# data_valid = data_valid.map(_parse).batch(BATCH_SIZE).repeat()





############################################################
#  Model definition
############################################################


# base_model = tf.keras.applications.ResNet50(include_top=False, pooling='avg')
# x = tf.keras.layers.Dense(1024, activation='relu')(base_model.output)
# x = tf.keras.layers.Dropout(0.25)(x)
# out = tf.keras.layers.Dense(14, kernel_regularizer=tf.keras.regularizers.l2(l=0.01))(x)

# model = tf.keras.Model(inputs=base_model.input, outputs=out)
#for layer in base_model.layers: layer.trainable = False


layers = [10, 16, 32, 64]
model = models.MultiTaskResNet(layers, 14, IMAGE_SIZE)

print(model.summary())

losses = {
    "mask_output": "binary_crossentropy",
    "landmarks_output": "mse"
}

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=losses,
              metrics=['accuracy']) 



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
    images_train,
    {"mask_output": masks_train, "landmarks_output": labels_train},
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(images_valid, {"mask_output": masks_valid, "landmarks_output": labels_valid})
)

# model.fit(
#     data_train,
#     epochs=EPOCHS,
#     steps_per_epoch=STEPS_PER_EPOCH,
#     validation_data=data_valid,
#     validation_steps=3,
#     callbacks=callbacks
#     )

#model.save_weights('../weights/landmarks/')

############################################################
#  Prediction/Testing
############################################################

filenames = os.listdir(PATH+'resized/')

test_images = np.array([sk.io.imread(PATH+'resized/' + filenames[i]) for i in range(2)])

test_images = np.array([sk.transform.resize(image, IMAGE_SIZE) for image in test_images])

masks, landmarks = model.predict(test_images)

for i in range(len(masks)):
    out = sk.transform.resize(masks[i], (500,500,3))
    sk.io.imsave('test' + str(i) + '.jpg', out)

print(landmarks)
