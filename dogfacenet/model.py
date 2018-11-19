"""
DogFaceNet
The main DogFaceNet implementation

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as KA
import tensorflow.keras.layers as KL

############################################################
#  NASNet Graph
############################################################

base_model = KA.NASNetMobile(input_shape=(224,224,3), include_top=False)

x = base_model.output
x = KL.GlobalAveragePooling2D()(x)
x = KL.Dense(1056, activation='relu')(x)
model = tf.keras.Model(inputs=base_model.input, outputs=x)


############################################################
#  Loss Functions
############################################################

def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)
        
        basic_loss = tf.add(tf.subtract(pos_dist,neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
      
    return loss