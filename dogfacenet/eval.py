"""
Evaluation of DogFaceNet model

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training import *

#----------------------------------------------------------------------------
# Config.

PATH        = '../data/dogfacenet/aligned/after_4_bis/' # Path to the directory of the saved dataset
PATH_SAVE   = '../output/history/'                      # Path to the directory where the history will be stored
PATH_MODEL  = '../output/model/2019.04.22 - best/'      # Path to the directory where the model will be stored
SIZE        = (224,224,3)                               # Size of the input images
TEST_SPLIT  = 0.1                                       # Train/test ratio

NET_NAME    = '2019.04.22.dogfacenet_v26'               # Network saved name
START_EPOCH = 282                                       # Start the training at a specified epoch

#----------------------------------------------------------------------------
# Import the dataset.

assert os.path.isdir(PATH), '[Error] Provided PATH for dataset does not exist.'

print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1
assert len(labels)!=0, '[Error] No data provided.'

print('Done.')

print('Total number of imported pictures: {:d}'.format(len(labels)))

nbof_classes = len(np.unique(labels))
print('Total number of classes: {:d}'.format(nbof_classes))

#----------------------------------------------------------------------------
# Split the dataset.

nbof_test = int(TEST_SPLIT*nbof_classes)

keep_test = np.less(labels,nbof_test)
keep_train = np.logical_not(keep_test)

filenames_test = filenames[keep_test]
labels_test = labels[keep_test]

filenames_train = filenames[keep_train]
labels_train = labels[keep_train]

print("Number of training data: " + str(len(filenames_train)))
print("Number of training classes: " + str(nbof_classes-nbof_test))
print("Number of testing data: " + str(len(filenames_test)))
print("Number of testing classes: " + str(nbof_test))

#----------------------------------------------------------------------------
# Loss definition.

alpha = 0.3
def triplet(y_true,y_pred):
    
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)
    
    return K.less(ap+alpha,an)

#----------------------------------------------------------------------------
# Model definition.

print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(PATH_MODEL,NET_NAME,START_EPOCH))

model = tf.keras.models.load_model(
    '{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,START_EPOCH),
    custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

print('Done.')

#----------------------------------------------------------------------------
# Verification task, create pairs

print('Verification task, pairs creation...')

NBOF_PAIRS = 5000
#NBOF_PAIRS = len(images_test)

# Create pairs
h,w,c = SIZE
pairs = []
issame = np.empty(NBOF_PAIRS)
class_test = np.unique(labels_test)
for i in range(NBOF_PAIRS):
    alea = np.random.rand()
    # Pair of different dogs
    if alea < 0.5:
        # Choose classes:
        class1 = np.random.randint(len(class_test))
        class2 = np.random.randint(len(class_test))
        while class1==class2:
            class2 = np.random.randint(len(class_test))
            
        # Extract images of this class:
        images_class1 = filenames_test[np.equal(labels_test,class1)]
        images_class2 = filenames_test[np.equal(labels_test,class2)]
        
        # Chose an image amoung these selected images
        pairs = pairs + [images_class1[np.random.randint(len(images_class1))]]
        pairs = pairs + [images_class2[np.random.randint(len(images_class2))]]
        issame[i] = 0
    # Pair of same dogs
    else:
        # Choose a class
        clas = np.random.randint(len(class_test))
        images_class = filenames_test[np.equal(labels_test,clas)]
        
        # Select two images from this class
        idx_image1 = np.random.randint(len(images_class))
        idx_image2 = np.random.randint(len(images_class))
        while idx_image1 == idx_image2:
            idx_image2 = np.random.randint(len(images_class))
        
        pairs = pairs + [images_class[idx_image1]]
        pairs = pairs + [images_class[idx_image2]]
        issame[i] = 1

print('Done.')

#----------------------------------------------------------------------------
# Verification task, evaluate the pairs

print('Verification task, model evaluation...')

predict=model.predict_generator(predict_generator(pairs, 32), steps=np.ceil(len(pairs)/32))
# Separates the pairs
emb1 = predict[0::2]
emb2 = predict[1::2]

# Computes distance between pairs
diff = np.square(emb1-emb2)
dist = np.sum(diff,1)


best = 0
best_t = 0
thresholds = np.arange(0.001,4,0.001)
for i in range(len(thresholds)):
    less = np.less(dist, thresholds[i])
    acc = np.logical_not(np.logical_xor(less, issame))
    acc = acc.astype(float)
    out = np.sum(acc)
    out = out/len(acc)
    if out > best:
        best_t = thresholds[i]
        best = out

print('Done.')
print("Best threshold: " + str(best_t))
print("Best accuracy: " + str(best))

# Test: Look at wrong pairs
t = 0.68
fa = []
fr = []
for i in range(len(dist)):
    # false accepted
    if issame[i] == 0 and dist[i]<t:
        fa += [i]
    # false rejected
    if issame[i] == 1 and dist[i]>t:
        fr += [i]

s = 10
sr = 20
n = 5
print('Ground truth: {:s}'.format(str(issame[s:(n+s)])))
fig = plt.figure(figsize=(11,2.8*n))
for i in range(s,s+n):
    # False accepted: columns 1 and 2
    plt.subplot(n,4,4*(i-s)+1)
    plt.imshow(load_images([pairs[2*fa[i+s]]])[0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(n,4,4*(i-s)+2)
    plt.imshow(load_images([pairs[2*fa[i+s]+1]])[0])
    plt.xticks([])
    plt.yticks([])
    # False rejected: columns 3 and 4
    plt.subplot(n,4,4*(i-s)+3)
    plt.imshow(load_images([pairs[2*fr[i+sr]]])[0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(n,4,4*(i-s)+4)
    plt.imshow(load_images([pairs[2*fr[i+sr]+1]])[0])
    plt.xticks([])
    plt.yticks([])

plt.show()

threshold = 0.3
less = np.less(dist, threshold)
acc = np.logical_not(np.logical_xor(less, issame))
acc = acc.astype(float)
out = np.sum(acc)
out = out/len(acc)

print("Accuracy: " + str(out))