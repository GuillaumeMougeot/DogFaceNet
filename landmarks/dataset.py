"""
DogFaceNet
Dataset retrieving for landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import numpy as np
import os
import skimage as sk
import pandas as pd
from ast import literal_eval # string to dict
import matplotlib.pyplot as plt
from tqdm import tqdm


############################################################
#  Data pre-processing for landmarks detection
############################################################


def get_landmarks_dataset(path='../data/landmarks/', split=0.8, output_shape=(500,500,3), save=True):
    """
    Gets images from the {path + 'images/'} directory, split the dataset in train (=0.8*dataset_size)
    and validation (=0.2*dataset_size). The size of the output images is defined by output_shape.
    If save=True then the resized images will be saved in {path + 'resized/'} directory.
    """
    csv_path = path
    for file in os.listdir(path):
        if '.csv' in file:
            csv_path += file
    df = pd.read_csv(csv_path)

    index = df.index
    
    filenames = df.loc[:,'filename']
    dictionary = [literal_eval(df.loc[:,'region_shape_attributes'][i]) for i in range(len(index))]

    h,w,c = output_shape
    images = np.empty((0,h,w,c))
    labels = np.empty((0,7,2))

    print("Resizing images...")
    for i in tqdm(range(0,len(filenames),7)):
        image = sk.io.imread(path + 'images/' + filenames[i])
        
        if len(image.shape)>1:
            image_resized = sk.transform.resize(image, output_shape, mode='reflect', anti_aliasing=False)

            images = np.append(images, np.expand_dims(image_resized, axis=0), axis=0)

            if save:
                sk.io.imsave(path + 'resized/' + filenames[i], image_resized)

            x, y, _ = image.shape
            a = h/x
            b = w/y

            landmarks = np.empty((7,2))
            for j in range(7):
                landmarks[j] = np.array([
                    dictionary[i + j]['cx'] * b,
                    dictionary[i + j]['cy'] * a
                    ])
            
            labels = np.append(labels, np.expand_dims(landmarks, axis=0), axis=0)

    print("Done.")

    assert len(images)==len(labels)

    train_split = int(0.8*len(images))

    if save:
        np.save(path + 'resized_labels.npy', labels)

    return images[:train_split], labels[:train_split], images[train_split:], labels[train_split:]

if __name__=="__main__":
    get_landmarks_dataset()

    