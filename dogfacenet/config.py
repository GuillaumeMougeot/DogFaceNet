"""
DogFaceNet
Base Configuration class

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import numpy as np

# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.
class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name of the configuration
    NAME = None

    ARCHITECTURE = "resnet101"

    IMAGE_RESIZE_MODE = "square"
    
    IMAGE_CHANNEL_COUNT = 3

    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
