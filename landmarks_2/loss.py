import numpy as np
import tensorflow as tf

import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

#----------------------------------------------------------------------------
# Mean square error.

def mse(N, reals, labels, is_training=True):
    predictions = N.get_output_for(reals, is_training=is_training)
    return tf.losses.mean_squared_error(labels, predictions, weights=100.)

#----------------------------------------------------------------------------
# Focal loss.

def focal_loss(N, reals, gt_outputs, is_training=True, alpha=0.25, gamma=2):
    pred = N.get_output_for(reals, is_training=is_training)
    assert gt_outputs.shape[-1] == pred.shape[-1], '[{:10s}] Prediction and ground truth shapes do not match: GT {:}, PRED {:}'.format('Error', gt_outputs.shape, pred.shape)
    loss = - alpha * (
        gt_outputs * tf.pow(1-pred,gamma) * tf.log(pred) + \
        (1-gt_outputs) * tf.pow(pred,gamma) * tf.log(1-pred))
    return tf.math.reduce_sum(loss)

def sigmoid_focal_loss_2(N, reals, gt_outputs, is_training=True, alpha=0.25):
    pred = N.get_output_for(reals, is_training=is_training)
    assert gt_outputs.shape[-1] == pred.shape[-1], '[{:10s}] Prediction and ground truth shapes do not match: GT {:}, PRED {:}'.format('Error', gt_outputs.shape, pred.shape)
    exp_1 = 1+tf.exp(-pred)
    loss = alpha / tf.square(exp_1) * ((gt_outputs * (tf.exp(-2*pred)-1)+1)*tf.log(exp_1)+pred*(1-gt_outputs))
    return tf.math.reduce_sum(loss)

#----------------------------------------------------------------------------
# Focal loss with refinement boxes

def sigmoid_focal_loss_2_ref(N, reals, gt_outputs, gt_ref, is_training=True, alpha=0.25):
    pred, pred_ref = N.get_output_for(reals, is_training=is_training)
    
    # Focal loss
    assert gt_outputs.shape[-1] == pred.shape[-1], '[{:10s}] Prediction and ground truth shapes do not match: GT {:}, PRED {:}'.format('Error', gt_outputs.shape, pred.shape)
    exp_1 = 1+tf.exp(-pred)
    focalLoss = alpha / tf.square(exp_1) * ((gt_outputs * (tf.exp(-2*pred)-1)+1)*tf.log(exp_1)+pred*(1-gt_outputs))
    focalLoss = tf.math.reduce_sum(focalLoss)
    # pred = tf.nn.sigmoid(pred)
    # focalLoss = - alpha * gt_outputs * tf.square(1-pred) * tf.log(pred) - alpha * (1-gt_outputs) * tf.square(pred) * tf.log(1-pred)
    # focalLoss = tf.math.reduce_sum(focalLoss)

    # Refinement loss
    ## L2 Loss
    # mask = (gt_outputs > 0.)
    # pred_ref = tf.boolean_mask(pred_ref, mask)
    # gt_ref = tf.boolean_mask(gt_ref, mask)
    # refLoss = tf.math.reduce_sum(tf.square(pred_ref-gt_ref))

    ## Smooth L1 Loss 
    mask = (gt_outputs > 0.)
    pred_ref = tf.boolean_mask(pred_ref, mask)
    gt_ref = tf.boolean_mask(gt_ref, mask)
    refLoss = tf.where(tf.math.abs(pred_ref-gt_ref)>1.,x=tf.math.abs(pred_ref-gt_ref),y=tf.square(pred_ref-gt_ref))
    refLoss = tf.math.reduce_sum(refLoss)
    # refLoss = tf.math.reduce_sum(tf.square(pred_ref-gt_ref))

    return focalLoss + refLoss