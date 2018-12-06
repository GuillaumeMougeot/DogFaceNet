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


def resize_dataset(path='../data/landmarks/', output_shape=(500,500,3)):
    """
    Resize images from the {path + 'images/'} directory and the labels from
    the csv file in the {path} directory.
    The size of the output images is defined by output_shape.
    Then the resized images will be saved in {path + 'resized/'} directory
    and the resized labels in {path + 'resized_labels.npy'}
    """
    csv_path = path
    for file in os.listdir(path):
        if '.csv' in file:
            csv_path += file
    df = pd.read_csv(csv_path)

    index = df.index
    
    filenames = df.loc[:,'filename']
    dictionary = [literal_eval(df.loc[:,'region_shape_attributes'][i]) for i in range(len(index))]

    h,w,_ = output_shape
    labels = np.empty((0,7,2))
    
    print("Resizing images...")
    for i in tqdm(range(0,len(filenames)-7,7)):
        image = sk.io.imread(path + 'images/' + filenames[i])
        
        if len(image.shape)>1:
            image_resized = sk.transform.resize(image, output_shape, mode='reflect', anti_aliasing=False)

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
        
    np.save(path + 'resized_labels.npy', labels)
    print("Done.")

    
def re_resize_dataset(path='../data/landmarks/', output_shape=(100,100,3)):
    """
    Re-resize the dataset after resize_dataset function
    """
    labels = np.load(path+'resized_labels.npy')
    h,w,_ = output_shape
    filenames = os.listdir(path+'resized/')
    output_labels = np.copy(labels)
    for i in tqdm(range(len(filenames))):
        image = sk.io.imread(path+'resized/'+filenames[i])
        
        x, y, _ = image.shape
        a = h/x
        b = w/y
        for j in range(7):
            output_labels[i][j] = np.array([labels[i][j][0] * a, labels[i][j][1] * b])

        image_resized = sk.transform.resize(image, output_shape, mode='reflect', anti_aliasing=False)
        sk.io.imsave(path + 're_resized/' + filenames[i], image_resized)
    np.save(path + 're_resized_labels.npy', output_labels)


# Too slow...
def get_resized_dataset(path='../data/landmarks/', split=0.8, shape=(500,500,3)):
    labels = np.load(path+'resized_labels.npy')
    h,w,c = shape
    images = np.empty((0,h,w,c))


    csv_path = path
    for file in os.listdir(path):
        if '.csv' in file:
            csv_path += file
    df = pd.read_csv(csv_path)

    filenames = df.loc[:,'filename']

    print("Getting images...")
    for i in tqdm(range(0,len(filenames)-7,7)):
        image = sk.io.imread(path + 'resized/' + filenames[i])
        if len(image.shape)>1:
            images = np.append(images, np.expand_dims(image, axis=0), axis=0)
    print("Done.")

    assert len(images)==len(labels)

    train_split = int(split*len(images))

    return images[:train_split], labels[:train_split], images[train_split:], labels[train_split:]


if __name__=="__main__":
    #resize_dataset(output_shape=(100,100,3))
    re_resize_dataset(output_shape=(100,100,3))
    #train_images, train_labels, valid_images, valid_labels = get_resized_dataset()

    