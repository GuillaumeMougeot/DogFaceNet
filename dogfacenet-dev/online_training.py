"""
DogFaceNet
Functions for training on bigger datasets then offline_training module.
It does not load all the dataset into memory but just a part of it.
It mainly relies on keras data generators.
It contains:
 - Offline triplet generator: for soft and hard triplets
 - Online triplet generator: for soft and hard triplets

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import pickle
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from offline_training import *
from math import isnan

SIZE = (224,224,3)

def define_triplets_batch(filenames,labels,nbof_triplet = 21 * 3):
    """
    Generates offline soft triplet.
    Given a list of file names of pictures, their specific label and
    a number of triplet images, returns an array of triplet of images
    and their specific labels.

    Args:
     - filenames: array of strings. List of file names of the pictures. 
     - labels: array of integers.
     - nbof_triplet: integer. Has to be a multiple of 3.
     
     Returns:
     - triplet_train: array of pictures --> a 4D array. 
     - y_triplet: array of integers of same dimension as the first
     dimension of triplet_train. Contains the labels of the pictures.
    """
    triplet_train = []
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in range(0,nbof_triplet,3):
        # Pick a class and chose two pictures from this class
        classAP = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classAP)
        keep_classAP = filenames[keep]
        keep_classAP_idx = labels[keep]
        idx_image1 = np.random.randint(len(keep_classAP))
        idx_image2 = np.random.randint(len(keep_classAP))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(keep_classAP))

        triplet_train += [keep_classAP[idx_image1]]
        triplet_train += [keep_classAP[idx_image2]]
        y_triplet[i] = keep_classAP_idx[idx_image1]
        y_triplet[i+1] = keep_classAP_idx[idx_image2]
        # Pick a class for the negative picture
        classN = classes[np.random.randint(len(classes))]
        while classN==classAP:
            classN = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classN)
        keep_classN = filenames[keep]
        keep_classN_idx = labels[keep]
        idx_image3 = np.random.randint(len(keep_classN))
        triplet_train += [keep_classN[idx_image3]]
        y_triplet[i+2] = keep_classN_idx[idx_image3]
        
    return triplet_train, y_triplet

def define_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
    """
    [DEPRECATED] Use define_adaptive_hard_triplets_batch instead!
    Generates hard triplet for offline selection. It will consider the whole dataset.
    
    Args:
        -images: images from which the triplets will be created
        -labels: labels of the images
        -predict: predicted embeddings for the images by the trained model
        -alpha: threshold of the triplet loss
    Returns:
        -triplet
        -y_triplet: labels of the triplets
    """
    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    triplets = []
    y_triplets = np.empty(nbof_triplet)
    
    for i in range(0,nbof_triplet,3):
        # Chooses the first class randomly
        keep = np.equal(labels,classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        idx_image1 = np.random.randint(len(keep_labels))
        
        
        # Computes the distance between the chosen image and the rest of the class
        if use_pos:
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

            idx_image2 = np.argmax(dist_class)
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            i = 0
            while idx_image1==idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                i += 1
                if i == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        
        
        # Computes the distance between the chosen image and the rest of the other classes
        not_keep = np.logical_not(keep)
        
        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]

    #return triplets, y_triplets
    return np.array(triplets), y_triplets

def define_adaptive_hard_triplets_batch(filenames,labels,predict,nbof_triplet=21*3, use_neg=True, use_pos=True):
    """
    Generates hard triplet for offline selection. It will consider the whole dataset.
    This function will also return the predicted values.
    
    Args:
        -images: images from which the triplets will be created
        -labels: labels of the images
        -predict: predicted embeddings for the images by the trained model
        -alpha: threshold of the triplet loss
    Returns:
        -triplets
        -y_triplets: labels of the triplets
        -pred_triplets: predicted embeddings of the triplets
    """
    # Check if we have the right number of triplets
    assert nbof_triplet%3 == 0
    
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    
    triplets = []
    y_triplets = np.empty(nbof_triplet)
    pred_triplets = np.empty((nbof_triplet,predict.shape[-1]))
    
    for i in range(0,nbof_triplet,3):
        # Chooses the first class randomly
        keep = np.equal(labels,classes[np.random.randint(len(classes))])
        keep_filenames = filenames[keep]
        keep_labels = labels[keep]
        
        # Chooses the first image among this class randomly
        idx_image1 = np.random.randint(len(keep_labels))
        
        
        # Computes the distance between the chosen image and the rest of the class
        if use_pos:
            dist_class = np.sum(np.square(predict[keep]-predict[keep][idx_image1]),axis=-1)

            idx_image2 = np.argmax(dist_class)
        else:
            idx_image2 = np.random.randint(len(keep_labels))
            j = 0
            while idx_image1==idx_image2:
                idx_image2 = np.random.randint(len(keep_labels))
                # Just to prevent endless loop:
                j += 1
                if j == 1000:
                    print("[Error: define_hard_triplets_batch] Endless loop.")
                    break
        
        triplets += [keep_filenames[idx_image1]]
        y_triplets[i] = keep_labels[idx_image1]
        pred_triplets[i] = predict[keep][idx_image1]
        triplets += [keep_filenames[idx_image2]]
        y_triplets[i+1] = keep_labels[idx_image2]
        pred_triplets[i+1] = predict[keep][idx_image2]
        
        # Computes the distance between the chosen image and the rest of the other classes
        not_keep = np.logical_not(keep)
        
        if use_neg:
            dist_other = np.sum(np.square(predict[not_keep]-predict[keep][idx_image1]),axis=-1)
            idx_image3 = np.argmin(dist_other) 
        else:
            idx_image3 = np.random.randint(len(filenames[not_keep]))
            
        triplets += [filenames[not_keep][idx_image3]]
        y_triplets[i+2] = labels[not_keep][idx_image3]
        pred_triplets[i+2] = predict[not_keep][idx_image3]

    return np.array(triplets), y_triplets, pred_triplets

def load_images(filenames):
    """
    Use scikit-image library to load the pictures from files to numpy array.
    """
    h,w,c = SIZE
    images = np.empty((len(filenames),h,w,c))
    for i,f in enumerate(filenames):
        images[i] = sk.io.imread(f)/255.0
    return images

def image_generator(filenames, labels, batch_size=63, use_aug=True, datagen=datagen):
    """
    Training generator for soft triplets.
    """
    while True:
        f_triplet, y_triplet = define_triplets_batch(filenames, labels, batch_size)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def hard_image_generator(filenames, labels, predict, batch_size=63, use_neg=True, use_pos=True, use_aug=True, datagen=datagen):
    """
    Training generator for offline hard triplets.
    """
    while True:
        f_triplet, y_triplet = define_hard_triplets_batch(filenames, labels, predict, batch_size, use_neg=use_neg, use_pos=use_pos)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def predict_generator(filenames, batch_size=32):
    """
    Prediction generator.
    """
    for i in range(0,len(filenames),batch_size):
        images_batch = load_images(filenames[i:i+batch_size])
        yield images_batch

def online_hard_image_generator(
    filenames,
    labels,
    model,
    batch_size=63,
    nbof_subclasses=10,
    use_neg=True,
    use_pos=True,
    use_aug=True,
    datagen=datagen):
    """
    Generator to select online hard triplets for training.
    
    Arguments:
        -filenames
        -labels
    """
    while True:
        # Select a certain amount of subclasses
        classes = np.unique(labels)
        subclasses = np.random.choice(classes,size=nbof_subclasses,replace=False)
        
        keep_classes = np.equal(labels,subclasses[0])
        for i in range(1,len(subclasses)):
            keep_classes = np.logical_or(keep_classes,np.equal(labels,subclasses[i]))
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]
        predict = model.predict_generator(predict_generator(subfilenames, 32),
                                          steps=np.ceil(len(subfilenames)/32))
        
        f_triplet, y_triplet = define_hard_triplets_batch(subfilenames, sublabels, predict, batch_size, use_neg=use_neg, use_pos=use_pos)
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
        yield (i_triplet, y_triplet)

def online_adaptive_hard_image_generator(
    filenames,                  # Absolute path of the images
    labels,                     # Labels of the images
    model,                      # A keras model
    loss,                       # Current loss of the model
    batch_size      =63,        # Batch size (has to be a multiple of 3 for dogfacenet)
    nbof_subclasses =10,        # Number of subclasses from which the triplets will be selected
    use_aug         =True,      # Use data augmentation?
    datagen         =datagen):  # Data augmentation parameter
    """
    Generator to select online hard triplets for training.
    Include an adaptive control on the number of hard triplets included during the training.
    """
    
    hard_triplet_ratio = 0
    nbof_hard_triplets = 0
    while True:
        # Select a certain amount of subclasses
        classes = np.unique(labels)
        # In order to limit the number of computation for prediction,
        # we will not computes nbof_subclasses predictions for the hard triplets generation,
        # but int(nbof_subclasses*hard_triplet_ratio)+2, which means that the higher the
        # accuracy is the more prediction are going to be computed.
        subclasses = np.random.choice(classes,size=int(nbof_subclasses*hard_triplet_ratio)+2,replace=False)
        
        keep_classes = np.equal(labels,subclasses[0])
        for i in range(1,len(subclasses)):
            keep_classes = np.logical_or(keep_classes,np.equal(labels,subclasses[i]))
        subfilenames = filenames[keep_classes]
        sublabels = labels[keep_classes]
        predict = model.predict_generator(predict_generator(subfilenames, 32),
                                          steps=int(np.ceil(len(subfilenames)/32)))
        
        
        
        
        
        f_triplet_hard, y_triplet_hard, predict_hard = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, nbof_hard_triplets*3, use_neg=True, use_pos=True)
        f_triplet_soft, y_triplet_soft, predict_soft = define_adaptive_hard_triplets_batch(subfilenames, sublabels, predict, batch_size-nbof_hard_triplets*3, use_neg=False, use_pos=False)

        f_triplet = np.append(f_triplet_hard,f_triplet_soft)
        y_triplet = np.append(y_triplet_hard,y_triplet_soft)

        predict = np.append(predict_hard, predict_soft, axis=0)
        
        # Proportion of hard triplets in the generated batch
        #hard_triplet_ratio = max(0,1.2/(1+np.exp(-10*acc+5.3))-0.19)
        hard_triplet_ratio = np.exp(-loss * 10 / batch_size)

        if isnan(hard_triplet_ratio):
            hard_triplet_ratio = 0
        nbof_hard_triplets = int(batch_size//3 * hard_triplet_ratio)
        
        i_triplet = load_images(f_triplet)
        if use_aug:
            i_triplet = apply_transform(i_triplet, datagen)
            
        # Potential modif for different losses: re-labels the dataset from 0 to nbof_subclasses
        # dict_subclass = {subclasses[i]:i for i in range(nbof_subclasses)}
        # ridx_y_triplet = [dict_subclass[y_triplet[i]] for i in range(len(y_triplet))]
        
        yield (i_triplet, y_triplet)


