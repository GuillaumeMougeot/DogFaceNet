"""
DogFaceNet
Losses for dog identification and embeddings optimization:
    -arcface_loss: defined in https://arxiv.org/abs/1801.07698
                   implemented in https://github.com/auroua/InsightFace_TF
    -triplet_loss: defined in https://arxiv.org/abs/1503.03832
                   implemented in https://github.com/davidsandberg/facenet

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import math
import numpy as np

import tensorflow as tf

def arcface(out_num):

    def loss(embedding, labels, out_num=out_num, w_init=None, s=64., m=0.5):
        '''
        :param embedding: the input embedding vectors
        :param labels:  the input labels, the shape should be eg: (batch_size, 1)
        :param s: scalar value default is 64
        :param out_num: output class num
        :param m: the margin value, default is 0.5
        :return: the final cacualted output, this output is send into the tf.nn.softmax directly
        '''
        cos_m = math.cos(m)
        sin_m = math.sin(m)
        mm = sin_m * m  # issue 1
        threshold = math.cos(math.pi - m)
        with tf.variable_scope('loss'):
            # inputs and weights norm
            embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
            embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
            weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                    initializer=w_init, dtype=tf.float32)
            weights_norm = tf.norm(weights, axis=0, keepdims=True)
            weights = tf.div(weights, weights_norm, name='norm_weights')
            # cos(theta+m)
            cos_t = tf.matmul(embedding, weights, name='cos_t')
            cos_t2 = tf.square(cos_t, name='cos_2')
            sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
            sin_t = tf.sqrt(sin_t2, name='sin_t')
            cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

            # this condition controls the theta+m should be in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s*(cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)

            mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
            # mask = tf.squeeze(mask, 1)
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

            output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
        return output

    return loss


def arcface_loss(embedding, labels, out_num, w_init=None, s=64., m=0.5):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope('loss'):
        # inputs and weights norm
        embedding_norm = tf.norm(embedding, axis=1, keepdims=True)
        embedding = tf.div(embedding, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                initializer=w_init, dtype=tf.float32)
        weights_norm = tf.norm(weights, axis=0, keepdims=True)
        weights = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        cos_t = tf.matmul(embedding, weights, name='cos_t')
        cos_t2 = tf.square(cos_t, name='cos_2')
        sin_t2 = tf.subtract(1., cos_t2, name='sin_2')
        sin_t = tf.sqrt(sin_t2, name='sin_t')
        cos_mt = s * tf.subtract(tf.multiply(cos_t, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')

        # this condition controls the theta+m should be in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_t - threshold
        cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

        keep_val = s*(cos_t - mm)
        cos_mt_temp = tf.where(cond, cos_mt, keep_val)

        mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
        # mask = tf.squeeze(mask, 1)
        inv_mask = tf.subtract(1., mask, name='inverse_mask')

        s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

        output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='arcface_loss_output')
    return output



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


def deviation_loss(dict_pred):
    sum_class_loss = 0
    classes_loss = 0

    class_pred = {}

    # Compute all center of mass
    for _, (label, pred) in dict_pred:
        if label in class_pred.keys():
            class_pred[label][0] += pred
            class_pred[label][1] += 1
        else:
            class_pred[label] = (pred,1)
    for label in class_pred:
            class_pred[label][0] /= class_pred[label][1]

    # Compute all classes center of mass
    class_pred_values = np.array(class_pred.values())
    classes_center = np.sum(class_pred_values)/len(class_pred)
    classes_loss -= np.sum(np.log(np.linalg.norm(class_pred_values - classes_center)))
    
    # Compute 
    for _, (label, pred) in dict_pred:
        sum_class_loss += np.linalg.norm(pred - class_pred[label])

    return classes_loss + sum_class_loss