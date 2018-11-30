"""
DogFaceNet
A serie of standard models:
    -SimpleMLP: a dummy model with just a pool layer and two dense layers
    -Dummy_softmax: a dummy model for testing with a softmax output
    -Dummy_embedding: a dummy model with a embedding output
    -ResNet_embedding: the classic ResNet50 with embedding output
    -NasNet_embedding: reuse of the pretrained NasNetMobile built in
                       keras-application with an embedding output

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

import tensorflow as tf

class SimpleMLP(tf.keras.Model):

    def __init__(self, use_bn=False, use_dp=False, num_classes=10):
        super(SimpleMLP, self).__init__(name='mlp')
        self.use_bn = use_bn
        self.use_dp = use_dp
        self.num_classes = num_classes

        self.pool = tf.keras.layers.MaxPooling2D((7, 7))
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(32, activation='relu')
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')
        if self.use_dp:
            self.dp = tf.keras.layers.Dropout(0.5)
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization(axis=-1)

    def call(self, inputs):
        x = self.pool(inputs)
        x = self.flat(x)
        x = self.dense1(x)
        if self.use_dp:
            x = self.dp(x)
        if self.use_bn:
            x = self.bn(x)
        return self.dense2(x)

class Dummy_softmax(tf.keras.Model):
    def __init__(self, num_output):
        super(Dummy_softmax, self).__init__(name='dummy')
        self.conv1 = tf.keras.layers.Conv2D(10,(3, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(20,(3, 3))
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(40,(3, 3))
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(80,(3, 3))
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.layers.Dense(num_output, activation='softmax')
    
    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.avg_pool(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.dense(x)

        return x

class Dummy_embedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(Dummy_embedding, self).__init__(name='dummy')
        self.conv1 = tf.keras.layers.Conv2D(10,(3, 3))
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv2 = tf.keras.layers.Conv2D(20,(3, 3))
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv3 = tf.keras.layers.Conv2D(40,(3, 3))
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))
        self.conv4 = tf.keras.layers.Conv2D(80,(3, 3))
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense = tf.layers.Dense(emb_size)
    
    def __call__(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.avg_pool(x)
        x = self.dense(x)

        return tf.nn.l2_normalize(x)

class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)

class ResnetConvBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn1 = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv1(input_tensor)
        shortcut = self.bn1(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)

class ResNet_embedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(ResNet_embedding, self).__init__(name='resnet')
        self.conv1_pad = tf.keras.layers.ZeroPadding2D(padding=(3,3))
        self.conv1 = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2))
        self.bn_conv1 = tf.keras.layers.BatchNormalization()

        self.pool1_pad = tf.keras.layers.ZeroPadding2D(padding=(1,1))
        self.pool1 = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2))

        #filters = [[64,64,256], [128,128,512], [256,256,1024], [512,512,2048]]
        #nrof_identity_block = [2,3,5,2]
        filters = [[64,64,256]]
        nrof_identity_block = [1]

        self.in_layers = []
        for i in range(len(filters)):
            self.in_layers += [ResnetConvBlock(3, filters[i])]
            for _ in range(nrof_identity_block[i]):
                self.in_layers += [ResnetIdentityBlock(3, filters[i])]
        
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.embedding = tf.keras.layers.Dense(emb_size)

    def __call__(self, input_tensor=None, training=False):
        x = self.conv1_pad(input_tensor)
        x = self.conv1(x)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1_pad(x)
        x = self.pool1(x)

        for in_layer in self.in_layers:
            x = in_layer(x, training=training)
        
        x = self.avg_pool(x)
        x = self.embedding(x)

        return tf.nn.l2_normalize(x)

class NASNet_embedding(tf.keras.Model):
    def __init__(self, emb_size):
        super(NASNet_embedding, self).__init__(name='')

        self.pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_1 = tf.layers.Dense(1056, activation='relu')
        self.dropout = tf.layers.Dropout(0.5)
        self.dense_2 = tf.layers.Dense(emb_size)

    def __call__(self, input_tensor, input_shape=(224, 224, 3), training=True, unfreeze=True):
        # base_model = tf.keras.applications.NASNetMobile(
        #         input_tensor=input_tensor,
        #         input_shape=input_shape,
        #         include_top=False
        #         )

        # for layer in base_model.layers: layer.trainable = False
        # x = self.pool(base_model.output)
        x = self.pool(input_tensor)
        x = self.dense_1(x)
        if training:
            x = self.dropout(x)
        x = self.dense_2(x)

        return tf.keras.backend.l2_normalize(x)
