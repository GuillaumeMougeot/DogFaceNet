"""
DogFaceNet
A serie of standard models for landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import tensorflow as tf


def ConvNet(layers, num_output=14, input_shape=(500,500,3,), weight=None):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(layers[0], (3,3), strides=(2,2), activation='relu')(inputs)
    for i in range(1,len(layers)):
        x = tf.keras.layers.Conv2D(layers[i], (3,3), strides=(2,2), activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_output)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if weight!=None:
        model.load_weights(weight)
    
    return model

def ConvBnNet(layers, num_output=14, input_shape=(500,500,3,), weight=None):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(layers[0], (5,5), padding='same')(inputs)
    for i in range(1,len(layers)):
        x = tf.keras.layers.Conv2D(layers[i], (3,3), strides=(2,2), activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_output)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if weight!=None:
        model.load_weights(weight)
    
    return model


def ResNet(layers, num_output=14, input_shape=(500,500,3,), weight=None):

    def ResBlock(input_tensor, filters):
        x = tf.keras.layers.Conv2D(filters, (3,3), activation='relu', padding='same')(input_tensor)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Concatenate()([input_tensor, x])

    
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(layers[0], (5,5), padding='same')(inputs)
    
    for i in range(1,len(layers)):
        x = tf.keras.layers.BatchNormalization()(x)
        x = ResBlock(x, layers[i])
        x = ResBlock(x, layers[i])
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_output)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if weight!=None:
        model.load_weights(weight)
    
    return model
    
    



############################################################
#  Archives
############################################################

"""
The following method doesn't work... why? I don't know yet
Apparently denses layers needs an input shape to be precized
but not the conv layers... doesn't make sense for me
"""
# class ConvNet(tf.keras.Model):
#     def __init__(self, layers, num_classes):
#         super(ConvNet, self).__init__(name='convnet')
#         self.num_classes = num_classes

#         self.convs = [tf.keras.layers.Conv2D(layers[i], (3,3), strides=(2,2), activation='relu') for i in range(len(layers))]

#         self.flat = tf.keras.layers.Flatten()
#         self.denses = [tf.keras.layers.Dense(64*i, activation='relu') for i in range(2,0,-1)]
#         self.out = tf.keras.layers.Dense(num_classes)
    
#     def call(self, inputs):
#         x = self.convs[0](inputs)
#         for i in range(1,len(self.convs)):
#             x = self.convs[i](x)
#         x = self.flat(x)
#         for i in range(len(self.denses)):
#             x = self.denses[i](x)
#         return self.out(x)
    
#     def compute_output_shape(self, input_shape):
#         shape = tf.TensorShape(input_shape).as_list()
#         output_shape = [shape[0], self.num_classes]
#         return tf.TensorShape(output_shape)

"""
And the following techniques neither (same reasons)
"""
# model = tf.keras.Sequential([
#     tf.keras.layers.Conv2D(i, (3, 3), strides=(2, 2), activation='relu') for i in layers
# ] + [
#     tf.keras.layers.Conv2D(400, (3, 3), activation='relu'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(64, activation='relu', input_shape=()),
#     tf.keras.layers.Dense(64, activation='relu'),
#     tf.keras.layers.Dense(14)
# ])



"""
The following is working but it is uselessly complicated
"""
# class ConvNet(object):
#     def __init__(self, layers, num_classes, input_shape=(500,500,3,)):
#         self.layers = layers
#         self.num_classes = num_classes
#         self.model = self.build_model(input_shape)

#     def build_model(self, input_shape=(500,500,3,)):
#         inputs = tf.keras.Input(shape=(500,500,3,))

#         x = tf.keras.layers.Conv2D(self.layers[0], (3,3), strides=(2,2), activation='relu')(inputs)
#         for i in range(1,len(self.layers)):
#             x = tf.keras.layers.Conv2D(self.layers[i], (3,3), strides=(2,2), activation='relu')(x)
#         x = tf.keras.layers.Flatten()(x)
#         x = tf.keras.layers.Dense(128, activation='relu')(x)
#         x = tf.keras.layers.Dense(64, activation='relu')(x)
#         outputs = tf.keras.layers.Dense(self.num_classes)(x)

#         return tf.keras.Model(inputs=inputs, outputs=outputs)