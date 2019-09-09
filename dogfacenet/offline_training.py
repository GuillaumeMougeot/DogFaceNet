"""
DogFaceNet
Functions for offline training.
The online_training module should be prefered instead of this one.
offline_training will load all the dataset into computer memory.
Even if the training is slighty faster the computer can rapidly
run out of memory.

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm_notebook


datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=8,
    zoom_range=0.1,
    fill_mode='nearest',
    channel_shift_range = 0.1
)

def single_apply_transform(image, datagen):
    """
    Apply a data preprocessing transformation to a single image
    Args:
        -image
        -ImageDataGenerator
    Return:
        -an image of the same shape of the input but transformed
    """
    image_exp = np.expand_dims(image,0)
    for x in datagen.flow(image_exp, batch_size=1):
        return x[0]

def apply_transform(images, datagen):
    """
    Apply a data preprocessing transformation to n images
    Args:
        -images
        -ImageDataGenerator
    Return:
        -images of the same shape of the inputs but transformed
    """
    for x in datagen.flow(images, batch_size=len(images), shuffle=False):
        return x

def define_triplets(images,labels,nbof_triplet = 10000 * 3, datagen=datagen):
    _,h,w,c = images.shape
    triplet_train = np.empty((nbof_triplet,h,w,c))
    y_triplet = np.empty(nbof_triplet)
    classes = np.unique(labels)
    for i in tqdm_notebook(range(0,nbof_triplet,3)):
        # Pick a class and chose two pictures from this class
        classAP = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classAP)
        keep_classAP = images[keep]
        keep_classAP_idx = labels[keep]
        idx_image1 = np.random.randint(len(keep_classAP))
        idx_image2 = np.random.randint(len(keep_classAP))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(keep_classAP))

        triplet_train[i] = single_apply_transform(keep_classAP[idx_image1],datagen)
        triplet_train[i+1] = single_apply_transform(keep_classAP[idx_image2],datagen)
        y_triplet[i] = keep_classAP_idx[idx_image1]
        y_triplet[i+1] = keep_classAP_idx[idx_image2]
        # Pick a class for the negative picture
        classN = classes[np.random.randint(len(classes))]
        while classN==classAP:
            classN = classes[np.random.randint(len(classes))]
        keep = np.equal(labels,classN)
        keep_classN = images[keep]
        keep_classN_idx = labels[keep]
        idx_image3 = np.random.randint(len(keep_classN))
        triplet_train[i+2] = single_apply_transform(keep_classN[idx_image3],datagen)
        y_triplet[i+2] = keep_classN_idx[idx_image3]
        
    return triplet_train, y_triplet


def shuffle_classes(images,labels):
    """
    Shuffles the classes
    """
    classes = np.unique(labels)
    np.random.shuffle(classes)
    
    shuffled_images = np.empty(images.shape)
    shuffled_labels = np.empty(labels.shape)
    idx = 0
    for i in range(len(classes)):
        keep_classes = np.equal(labels,classes[i])
        length = np.sum(keep_classes.astype(int))
        shuffled_labels[idx:idx+length] = labels[keep_classes]
        shuffled_images[idx:idx+length] = images[keep_classes]
        idx += length
    return shuffled_images, shuffled_labels

def global_define_hard_triplets(images,labels,predict,datagen=datagen):
    """
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
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    nbof_classes = len(classes)
    _,h,w,c = images.shape
    triplets = np.empty((3*len(predict),h,w,c))
    y_triplets = np.empty(3*len(predict))
    
    idx_triplets = 0
    idx_images = 0
    
    for i in range(nbof_classes):
        keep_class = np.equal(labels,classes[i])
        
        #predict_class = mask_class.dot(predict)
        predict_other = np.copy(predict)
        for j in range(len(predict)):
            if keep_class[j]:
                predict_other[j] += np.inf
        
        keep_predict_class = predict[keep_class]
        
        for j in range(len(keep_predict_class)):
            # Computes the distance between the current vector and the vectors in the class
            dist_class = np.sum(np.square(keep_predict_class-keep_predict_class[j]),axis=-1)
            
            # Add the anchor
            triplets[idx_triplets] = single_apply_transform(images[idx_images+j],datagen)
            y_triplets[idx_triplets] = labels[idx_images+j]
            
            # Add the hard positive
            triplets[idx_triplets+1] = single_apply_transform(images[idx_images+np.argmax(dist_class)],datagen)
            y_triplets[idx_triplets+1] = labels[idx_images+np.argmax(dist_class)]
            
            # Computes the distance between the current vector and the vectors of the others classes
            dist_other = np.sum(np.square(predict_other-keep_predict_class[j]),axis=-1)
            
            # Add the hard negative
            triplets[idx_triplets+2] = single_apply_transform(images[np.argmin(dist_other)],datagen)
            y_triplets[idx_triplets+2] = labels[np.argmin(dist_other)]
            
            idx_triplets += 3
        
        idx_images += len(keep_predict_class)
        
    return triplets, y_triplets


def define_hard_triplets(images,labels,predict,class_subset_size=10, add=100*3):
    """
    Generates hard triplet for offline selection
    
    Args:
        -images: images from which the triplets will be created
        -labels: labels of the images
        -predict: predicted embeddings for the images by the trained model
        -alpha: threshold of the triplet loss
        -class_subset_class: number of classes in a subset
        -
    Returns:
        -triplet
        -y_triplet: labels of the triplets
    """
    _,idx_classes = np.unique(labels,return_index=True)
    classes = labels[np.sort(idx_classes)]
    nbof_classes = len(classes)
    _,h,w,c = images.shape
    triplets = np.empty((3*len(predict)+add*(nbof_classes//class_subset_size + 1),h,w,c))
    y_triplets = np.empty(3*len(predict)+add*(nbof_classes//class_subset_size + 1))
    idx = 0
    for i in tqdm_notebook(range(0,len(classes),class_subset_size)):
        selected_classes = classes[i:i+class_subset_size]
        keep_classes = np.array([labels[j] in selected_classes for j in range(len(labels))])
        
        selected_predict = predict[keep_classes]
        length = len(selected_predict)*3
        
        triplets_tmp,y_triplets_tmp = global_define_hard_triplets(
                                                                images[keep_classes],
                                                                labels[keep_classes],
                                                                selected_predict
                                                            )
        print(len(triplets_tmp))
        
        triplets[idx:idx+length] = triplets_tmp
        y_triplets[idx:idx+length] = y_triplets_tmp
        
        triplets[idx+length:idx+length+add], y_triplets[idx+length:idx+length+add] = define_triplets(images, labels, add)
        
        idx += len(triplets_tmp) + add
    return triplets, y_triplets