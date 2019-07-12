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

#----------------------------------------------------------------------------
