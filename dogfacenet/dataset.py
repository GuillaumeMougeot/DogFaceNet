"""
DogFaceNet
Dataset retrieving

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import numpy as np
import os
import skimage as sk
from tqdm import tqdm


def get_dataset(path='../data/full/', train_split=0.8):
    """ Get dataset
    path: the data folder, composed of a 'bg' folder and many others 'dog$index$' folders
    train_split: the proportion of data per class to keep for training
    """

    filenames_train = []
    labels_train = []

    filenames_valid = []
    labels_valid = []

    count_labels = 0

    for root, _, files in os.walk(path):
        n = len(files)
        if n>0:
            # In the beginning background dogs have all label 0
            if 'bg' in root:
                label = 0
                count_labels += n
            else:
                label = int(root[len(path) + 3:])
                count_labels += 1
            
            split = int(train_split * n)

            

            labels_train += [label for _ in range(split)]
            filenames_train += [root + '/' + files[i] for i in range(split)]

            labels_valid += [label for _ in range(n - split)]
            filenames_valid += [root + '/' + files[i] for i in range(split, n)]
    
    # Rename the background dogs
    count_train = max(labels_train) + 1
    for i in range(len(labels_train)):
        if labels_train[i] == 0:
            labels_train[i] = count_train
            count_train += 1

    count_valid = max(labels_valid) + 1
    for i in range(len(labels_valid)):
        if labels_valid[i] == 0:
            labels_valid[i] = count_valid
            count_valid += 1

    return np.array(filenames_train), np.array(labels_train), np.array(filenames_valid), np.array(labels_valid), count_labels

def resize_data(filenames_train, labels_train, filenames_valid, labels_valid, image_shape=[224,224,3]):
    x_train = np.empty([0] + image_shape)
    y_train = np.empty((0))

    x_valid = np.empty([0] + image_shape)
    y_valid = np.empty((0))

    for i in tqdm(range(len(filenames_train))):
        image = sk.io.imread(filenames_train[i])
        tqdm.write(str(np.shape(image)))
        if len(np.shape(image))==3:
            image_resized = sk.transform.resize(image, image_shape, mode='reflect')
            x_train = np.append(x_train, np.expand_dims(image_resized, axis=0), axis=0)
            y_train = np.append(y_train, labels_train[i])

    for i in tqdm(range(len(filenames_valid))):
        image = sk.io.imread(filenames_valid[i])
        if len(np.shape(image))==3:
            image_resized = sk.transform.resize(image, image_shape, mode='reflect')
            x_valid = np.append(x_valid, np.expand_dims(image_resized, axis=0), axis=0)
            y_valid = np.append(y_valid, labels_valid[i])
    return x_train, y_train, x_valid, y_valid

def save_resized_data(x_train, y_train, x_valid, y_valid, path='../data/resized/'):
    path_train = path + 'train/'
    for i in range(len(x_train)):
        sk.io.imsave(path_train + str(i) + '.jpg', x_train[i])
    np.save(path_train + 'labels.npy', y_train)

    path_valid = path + 'valid/'
    for i in range(len(x_valid)):
        sk.io.imsave(path_valid + str(i) + '.jpg', x_valid[i])
    np.save(path_valid + 'labels.npy', y_valid)

def get_resized_dataset(path='../data/resized/', image_shape=[224,224,3]):
    path_train = path + 'train/'

    x_train = np.empty([0] + image_shape)
    for file in tqdm(os.listdir(path_train)):
        if '.jpg' in file:
            image = sk.io.imread(path_train + file)
            x_train = np.append(x_train, np.expand_dims(image, axis=0), axis=0)
    
    y_train = np.load(path_train + 'labels.npy')

    path_valid = path + 'valid/'

    x_valid = np.empty([0] + image_shape)
    for file in tqdm(os.listdir(path_valid)):
        if '.jpg' in file:
            image = sk.io.imread(path_valid + file)
            x_valid = np.append(x_valid, np.expand_dims(image, axis=0), axis=0)
    
    y_valid = np.load(path_valid + 'labels.npy')

    return x_train, y_train, x_valid, y_valid

if __name__=='__main__':
    filenames_train, labels_train, filenames_valid, labels_valid,_ = get_dataset()
    x_train, y_train, x_valid, y_valid = resize_data(filenames_train, labels_train, filenames_valid, labels_valid)
    save_resized_data(x_train, y_train, x_valid, y_valid)

