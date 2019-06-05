# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

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
<<<<<<< HEAD
    return tf.losses.mean_squared_error(labels, predictions, weights=10.)
=======
    return tf.losses.mean_squared_error(labels, predictions, weights=100.)
>>>>>>> 2bc77daca12d4b9df42d0d1e04a923c8352cb731

#----------------------------------------------------------------------------
