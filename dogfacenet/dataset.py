"""
DogFaceNet
Dataset retrieving

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import numpy as np
import os


def get_dataset(path='../data/', train_split=0.8):
    """ Get dataset
    path: the data folder, composed of a 'bg' folder and many others 'dog$index$' folders
    train_split: the proportion of data per class to keep for training
    """

    filenames_train = []
    labels_train = []

    filenames_valid = []
    labels_valid = []
    for root, _, files in os.walk(path):
        n = len(files)
        if n>0:
            # In the beginning background dogs have all label 0
            if 'bg' in root:
                label = 0
            else:
                label = int(root[11:])
            
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

    return np.array(filenames_train), np.array(labels_train), np.array(filenames_valid), np.array(labels_valid)