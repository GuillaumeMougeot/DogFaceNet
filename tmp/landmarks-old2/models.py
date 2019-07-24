"""
DogFaceNet
A serie of standard models for landmarks detection

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import tensorflow as tf
import numpy as np

############################################################
#  "Standard" models
############################################################

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


def ResBlock(input_tensor, filters, strides=(2,2)):
    x = tf.keras.layers.Conv2D(filters, (3,3), padding='same')(input_tensor)
    x = tf.keras.layers.Conv2D(filters, (3,3), strides=strides, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    #x = tf.keras.layers.Dropout(0.25)(x)

    y = tf.keras.layers.MaxPool2D((1,1), strides=strides, padding='same')(input_tensor)
    return tf.keras.layers.Concatenate()([y, x])


def ResNet(layers, num_output=14, input_shape=(500,500,3,), weight=None):

    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(layers[0], (7,7), padding='same')(inputs)
    x = tf.keras.layers.Conv2D(layers[0], (3,3), activation='relu', padding='same') (x)

    for i in range(1,len(layers)):
        x = tf.keras.layers.BatchNormalization()(x)
        x = ResBlock(x, layers[i])
        x = ResBlock(x, layers[i])
    
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(
        num_output,
        kernel_regularizer=tf.keras.regularizers.l2(l=0.01)
        )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    if weight!=None:
        model.load_weights(weight)
    
    return model
    

############################################################
#  Models U-Net like training
############################################################


def TriNet(ratio=4, input_shape=(128,128,4)):
    """
    Arguments:
     -ratio: ratio of channel reduction in SE module
     -imput_shape: input image shape
    """
    
    def CBA_layer(x, filters, size=3, depth=2):
        
        for _ in range(depth):
            x = tf.keras.layers.Conv2D(filters, (size, size), padding='same') (x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Activation('relu')(x)
            
        return x
    
    def Res_layer(x, num_split, filters):
        '''
        ResNet-like layer
        '''
        
        # Spliting the branches and changing the size of the convolution
        splitted_branches = list()
        
        for i in range(num_split):
            if i+1 < 6:
                size = i+1 
            else:
                size = 3
            branch = CBA_layer(x, filters, size)
            splitted_branches.append(branch)
        
        x = tf.keras.layers.concatenate(splitted_branches)
        
        x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same') (x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        return x
    
    def SE_layer(x):
        '''
        SENet-like layer
        '''
        out_dim = int(np.shape(x)[-1])
        squeeze = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        excitation = tf.keras.layers.Dense(units=out_dim // ratio)(squeeze)
        excitation = tf.keras.layers.Activation('relu')(excitation)
        excitation = tf.keras.layers.Dense(units=out_dim)(excitation)
        excitation = tf.keras.layers.Activation('sigmoid')(excitation)
        excitation = tf.keras.layers.Reshape((1,1,out_dim))(excitation)
        
        scale = tf.keras.layers.multiply([x,excitation])
        
        return scale
    
    def RSE_layer(x, num_split, filters):
        r = Res_layer(x, num_split, filters)
        s = SE_layer(r)
        c = tf.keras.layers.concatenate([x,s])
        return tf.keras.layers.Activation('relu')(c)
    

    inputs = tf.keras.Input()

    s = tf.keras.layers.Lambda(lambda x: x / 255) (inputs)
    
    #Down 1
    r1 = RSE_layer(s, 3, 8)
    x = tf.keras.layers.MaxPooling2D((2, 2)) (r1)
    
    #Down 2
    r2 = RSE_layer(x, 4, 16)
    x = tf.keras.layers.MaxPooling2D((2, 2)) (r2)
    
    #Down 3
    r3 = RSE_layer(x, 6, 32)
    x = tf.keras.layers.MaxPooling2D((2, 2)) (r3)
    
    #Down 4
    r4 = RSE_layer(x, 8, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2)) (r4)
    
    #Down 5
    r5 = RSE_layer(x, 8, 128)
    x = tf.keras.layers.MaxPooling2D((2, 2)) (r5)
    
    #Middle
    x = RSE_layer(x, 6, 256)

    # First branch: landmarks detection
    y = RSE_layer(x, 4, 512)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    y = tf.keras.layers.Flatten()(y)
    y = tf.keras.layers.Dense(10)(y)
    landmarks_output = tf.keras.layers.Lambda(lambda x: x * 255, name='landmarks_output')(y)
    
    # Second branch: mask generation
    #Up 1
    x = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (x)
    x = tf.keras.layers.concatenate([x,r5])
    x = RSE_layer(x, 8, 128)
    
    #Up 2
    x = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (x)
    x = tf.keras.layers.concatenate([x,r4])
    x = RSE_layer(x, 8, 64)
    
    #Up 3
    x = tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (x)
    x = tf.keras.layers.concatenate([x,r3])
    x = RSE_layer(x, 6, 32)
    
    #Up 4
    x = tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (x)
    x = tf.keras.layers.concatenate([x,r2])
    x = RSE_layer(x, 4, 16)
    
    #Up 5
    x = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (x)
    x = tf.keras.layers.concatenate([x,r1])
    x = RSE_layer(x, 3, 8)
    
    mask_output = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask_output') (x)
    
    return tf.keras.Model(inputs, [landmarks_output, mask_output])


############################################################
#  Models for multi-task training
############################################################


def PNet(input_shape=(None,None,3,), alpha_bbox=0.5, alpha_landmarks=0.5):
    input_image = tf.keras.Input(shape=input_shape, name='input_image')
    input_beta = tf.keras.Input(shape=(1,), name='input_beta')

    s = tf.keras.layers.Lambda(lambda x: x / 255.0) (input_image)
    # x = tf.keras.layers.Conv2D(10,(3,3))(s)
    # x = tf.keras.layers.MaxPool2D((2,2))(x)
    x = ResBlock(s,16)
    x = ResBlock(x,32)

    branch_class = tf.keras.layers.Conv2D(1,(3,3), activation='relu', padding='same')(x)
    branch_class = tf.keras.layers.BatchNormalization()(branch_class)
    branch_class = tf.keras.layers.GlobalAveragePooling2D()(branch_class)
    branch_class = tf.keras.layers.Flatten()(branch_class)
    output_class = tf.keras.layers.Dense(1,activation='sigmoid',name='output_class')(branch_class)

    branch_bbox = tf.keras.layers.Conv2D(4, (3,3), activation='relu', padding='same')(x)
    branch_bbox = tf.keras.layers.BatchNormalization()(branch_bbox)
    branch_bbox = tf.keras.layers.GlobalAveragePooling2D()(branch_bbox)
    branch_bbox = tf.keras.layers.Flatten()(branch_bbox)
    #branch_bbox = tf.keras.layers.Dense(4)(branch_bbox)

    beta_bbox = tf.keras.layers.Lambda(lambda x: x*alpha_bbox)(input_beta)
    output_bbox = tf.keras.layers.multiply([
        beta_bbox,
        branch_bbox],
        name='output_bbox'
    )

    branch_landmarks = tf.keras.layers.Conv2D(10, (3,3), activation='relu', padding='same')(x)
    branch_landmarks = tf.keras.layers.BatchNormalization()(branch_landmarks)
    branch_landmarks = tf.keras.layers.GlobalAveragePooling2D()(branch_landmarks)
    branch_landmarks = tf.keras.layers.Flatten()(branch_landmarks)
    #branch_landmarks = tf.keras.layers.Dense(10)(branch_landmarks)

    beta_landmarks = tf.keras.layers.Lambda(lambda x: x*alpha_landmarks)(input_beta)
    output_landmarks = tf.keras.layers.multiply([
        beta_landmarks,
        branch_landmarks],
        name='output_landmarks'
    )

    model = tf.keras.Model(inputs=[input_image, input_beta], outputs=[output_class,output_bbox,output_landmarks])

    return model



def MultiTaskResNet(layers, num_output=10, input_shape=(500,500,3,)):

    inputs = tf.keras.Input(shape=input_shape)

    x = ResBlock(inputs, layers[0],strides=(1,1))
    for i in range(1, len(layers)-1):
        x = ResBlock(x, layers[i])

    # First output: the binary mask image
    mask_output = tf.keras.layers.Conv2D(1,3,activation='sigmoid',padding='same',name='mask_output')(x)



    x = ResBlock(x, layers[-1])

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)

    # Second output: the 10 facial landmarks
    landmarks_output = tf.keras.layers.Dense(num_output, name='landmarks_output')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[mask_output, landmarks_output])

    return model


############################################################
#  Archives
############################################################

"""
The following method doesn't work... why? I don't know yet
Answer: Apparently denses layers needs an input shape to be precized
but not the conv layers
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