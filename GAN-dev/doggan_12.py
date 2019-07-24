from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys

import numpy as np

import os
import skimage as sk
from tqdm import tqdm


PATH = '../data/dogfacenet/aligned/after_4_resized/'

PATH_SAVE = '../output/images/dcgan/dogs/'
PATH_MODEL = '../output/model/'
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 127
        self.img_cols = 127
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128

        # Bug fixed: if the generator and the discriminator have the same
        # optimizers it doesn't converge.
        # optimizer_d = Adam(0.0002, 0.5)
        # optimizer_c = Adam(0.001, 0.5)
        optimizer = Adam(0.0002,0.,0.99,1e-8,1e-5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def inception_block_down(self, inputs, output_filters=[64,64,64,64], kernels=[3,5,7], strides=1):
        assert len(kernels)==len(output_filters)-1, "[Error] Not the appropriate number of filters."

        x1 = Conv2D(output_filters[0], 1, use_bias=False)(inputs)
        x1 = MaxPooling2D(strides, padding='same')(x1)

        concat = [x1]

        for i in range(len(kernels)):
            x = Conv2D(output_filters[i+1], 1, use_bias=False)(inputs)
            x = Conv2D(output_filters[i+1], kernels[i], use_bias=False, strides=strides, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            concat += [x]

        return Concatenate()(concat)

    def inception_block_up(self, inputs, output_filters=[64,64,64,64], kernels=[3,5,7], strides=1):
        assert len(kernels)==len(output_filters)-1, "[Error] Not the appropriate number of filters."

        x1 = Conv2D(output_filters[0], 1, use_bias=False)(inputs)
        x1 = UpSampling2D(strides)(x1)

        concat = [x1]

        for i in range(len(kernels)):
            x = Conv2D(output_filters[i+1], 1, use_bias=False)(inputs)
            x = Conv2DTranspose(output_filters[i+1], kernels[i], use_bias=False, strides=strides, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            concat += [x]

        return Concatenate()(concat)

    def first_inception_block_down(self, inputs, output_filters=[64,64,64,64], kernels=[3,3,5,7], strides=1):
        assert len(kernels)==len(output_filters), "[Error] Not the appropriate number of filters: %i kernels and %i filters." % (len(kernels), len(output_filters))

        concat = []
        for i in range(len(kernels)):
            x = Conv2D(output_filters[i], 1, use_bias=False)(inputs)
            x = Conv2D(output_filters[i], kernels[i], use_bias=False, strides=strides)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            if kernels[i]>=3:
                x = Conv2D(output_filters[i], kernels[i]-2, use_bias=False)(x)
            concat += [x]

        x = Concatenate()(concat)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def first_inception_block_up(self, inputs, output_filters=[64,64,64,64], kernels=[3,3,5,7], strides=1):
        assert len(kernels)==len(output_filters), "[Error] Not the appropriate number of filters: %i kernels and %i filters." % (len(kernels), len(output_filters))

        concat = []
        for i in range(len(kernels)):
            x = Conv2D(output_filters[i], 1, use_bias=False)(inputs)
            x = Conv2DTranspose(output_filters[i], kernels[i], use_bias=False, strides=strides)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            if kernels[i]>=3:
                x = Conv2D(output_filters[i], kernels[i]-2, use_bias=False)(x)
            concat += [x]

        x = Concatenate()(concat)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))

        x = Reshape((1,1,self.latent_dim))(noise)
        
        x = self.first_inception_block_up(x, [64,64,32,16], [3,3,5,7], 2)
        x = self.inception_block_down(x, [64,64,32,16], [3,5,7], 1)

        for _ in range(4):
            x = self.first_inception_block_up(x, [64,64,32,16],[3,3,5,7], 2)
            x = self.inception_block_down(x, [64,64,64,32],[3,3,5], 1)

        x = self.first_inception_block_up(x, [64,64,32,16], [3,3,5,7], 2)
        x = self.inception_block_down(x, [64,64,32,16], [3,5,7], 1)

        x = Conv2D(self.channels,(3,3),padding='same')(x)
        img = Activation('tanh')(x)
        
        model = Model(noise, img)

        model.summary()

        return Model(noise, img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        x = self.inception_block_down(img,[32,32,16,8],[3,5,7], 2)
        x = self.inception_block_down(x,[32,32,16,8],[3,5,7], 2)
        x = self.inception_block_down(x,[64,64,32,8],[3,5,7], 2)
        x = self.inception_block_down(x,[64,64,64,32],[3,3,5], 2)
        x = self.inception_block_down(x,[128]*4,[3]*3, 2)
        x = self.first_inception_block_down(x,[64]*8,[3]*8, 1)
        x = AveragePooling2D()(x)
        x = Flatten()(x)
        validity = Dense(1, activation='sigmoid')(x)

        model = Model(img, validity)
        
        model.summary()

        return model

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        print("Load data into memory...")
        filenames = np.empty(0)
        idx = 0
        for root,_,files in os.walk(PATH):
            if len(files)>1:
                for i in range(len(files)):
                    files[i] = root + '/' + files[i]
                filenames = np.append(filenames,files)

        # max_size = 100
        max_size = len(filenames)
        X_train = np.empty((max_size,self.img_cols,self.img_rows,self.channels))
        for i,f in tqdm(enumerate(filenames)):
            if i == max_size:
                break
            X_train[i] = sk.io.imread(f)/ 127. - 1.
        print("done")


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                # axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(PATH_SAVE+"mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=20000, batch_size=4, save_interval=200)
