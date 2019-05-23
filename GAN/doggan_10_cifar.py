"""
DOG GAN for cifar 10.
"""

from __future__ import print_function, division

from keras.datasets import cifar10
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, AveragePooling2D
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
import keras.backend as K
from functools import partial

import matplotlib.pyplot as plt

import sys

import numpy as np

import os
import skimage as sk
from tqdm import tqdm


PATH = '../data/dogfacenet/aligned/after_4_resized_2/'

PATH_SAVE = '../output/images/dcgan/cifar10/'
PATH_MODEL = '../output/model/gan/cifar10/'
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1


class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class DCGAN():
    # def __init__(self, use_saved=False, filenames=None, is_eval=False):
    #     # Input shape
    #     self.img_rows = 32
    #     self.img_cols = 32
    #     self.channels = 3
    #     self.img_shape = (self.img_rows, self.img_cols, self.channels)
    #     self.latent_dim = 64

    #     # Bug fixed: if the generator and the discriminator have the same
    #     # optimizers it doesn't converge.
    #     # optimizer_d = Adam(0.0002, 0.5)
    #     # optimizer_c = Adam(0.001, 0.5)
    #     # optimizer = Adam(5e-5,0.2,0.99,1e-8)        
    #     optimizer = Adam(0.0002,0.,0.99,1e-8,1e-5)


    #     if is_eval==False:
    #         # Build and compile the discriminator
    #         self.discriminator = self.build_discriminator()

    #         if use_saved:
    #             assert filenames!=None, "[Error] A file name has to be defined to load a model."
    #             self.discriminator.load_weights(filenames[0])

    #         self.discriminator.compile(loss=wasserstein_loss,
    #             optimizer=optimizer,
    #             metrics=['accuracy'])

    #     # Build the generator
    #     self.generator = self.build_generator()
    #     if use_saved:
    #         self.generator.load_weights(filenames[1])

    #     # The generator takes noise as input and generates imgs
    #     z = Input(shape=(self.latent_dim,))
    #     img = self.generator(z)

    #     if is_eval==False:
    #         # For the combined model we will only train the generator
    #         self.discriminator.trainable = False

    #         # The discriminator takes generated images as input and determines validity
    #         valid = self.discriminator(img)

    #         # The combined model  (stacked generator and discriminator)
    #         # Trains the generator to fool the discriminator
        
    #         self.combined = Model(z, valid)
    #         self.combined.compile(loss=wasserstein_loss, optimizer=optimizer)

    def __init__(self, use_saved=False, filenames=None, is_eval=False):
        # Input shape
        self.img_rows = 32
        self.img_cols = 32
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64

        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
        self.generator = self.build_generator()
        self.critic = self.build_discriminator()

        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        # Generate images based of noise
        img = self.generator(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def inception_block_down(self, inputs, output_filters, strides=1, kernels=[3,5,7]):
        assert len(kernels)>0, "[Error] Too few kernels."
        nbof_kernels = len(kernels) + 1

        x1 = Conv2D(output_filters//nbof_kernels, 1, use_bias=False)(inputs)
        x1 = MaxPooling2D(strides, padding='same')(x1)

        concat = [x1]

        for i in range(nbof_kernels-1):
            x = Conv2D(output_filters//nbof_kernels, 1, use_bias=False)(inputs)
            x = Conv2D(output_filters//nbof_kernels, kernels[i], use_bias=False, strides=strides, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            concat += [x]

        return Concatenate()(concat)

    def inception_block_up(self, inputs, output_filters, strides=1, kernels=[3,5,7]):
        assert len(kernels)>0, "[Error] Too few kernels."
        nbof_kernels = len(kernels) + 1

        x1 = Conv2D(output_filters//nbof_kernels, 1, use_bias=False)(inputs)
        x1 = UpSampling2D(strides)(x1)

        concat = [x1]

        for i in range(nbof_kernels-1):
            x = Conv2D(output_filters//4, 1, use_bias=False)(inputs)
            x = Conv2DTranspose(output_filters//nbof_kernels, kernels[i], use_bias=False, strides=strides, padding='same')(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)
            concat += [x]

        return Concatenate()(concat)

    def first_inception_block_up(self, inputs, output_filters, strides=1, kernels=[3,3,5,7]):
        # x1 = Conv2D(output_filters//4, 1, use_bias=False)(inputs)
        # x1 = UpSampling2D(strides)(x1)

        assert len(kernels)>0, "[Error] Too few kernels."
        nbof_kernels = len(kernels)

        concat = []
        for i in range(nbof_kernels):
            x = Conv2D(output_filters//nbof_kernels, 1, use_bias=False)(inputs)
            x = Conv2DTranspose(output_filters//4, kernels[i], use_bias=False, strides=strides)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            if kernels[i]>=3:
                x = Conv2D(output_filters//nbof_kernels, kernels[i]-2, use_bias=False)(x)
            concat += [x]

        x = Concatenate()(concat)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def identity_inception_block_down(self, inputs, output_filters=[64,64,64,64], kernels=[3,5,7], strides=1):
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

    def identity_inception_block_up(self, inputs, output_filters=[64,64,64,64], kernels=[3,5,7], strides=1):
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

    def conv_inception_block_down(self, inputs, output_filters=[64,64,64,64], kernels=[3,3,5,7], strides=1, padding='same'):
        assert len(kernels)==len(output_filters), "[Error] Not the appropriate number of filters: %i kernels and %i filters." % (len(kernels), len(output_filters))

        concat = []
        for i in range(len(kernels)):
            x = Conv2D(output_filters[i], 1, use_bias=False)(inputs)
            x = Conv2D(output_filters[i], kernels[i], use_bias=False, strides=strides, padding=padding)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            # if kernels[i]>3:
            #     print(x.shape)
            #     x = Conv2D(output_filters[i], kernels[i]-2, use_bias=False)(x)
            #     x = BatchNormalization(momentum=0.8)(x)
            #     x = LeakyReLU(alpha=0.2)(x)
            concat += [x]
        
        x = Concatenate()(concat)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x

    def conv_inception_block_up(self, inputs, output_filters=[64,64,64,64], kernels=[3,3,5,7], strides=1, padding='same'):
        assert len(kernels)==len(output_filters), "[Error] Not the appropriate number of filters: %i kernels and %i filters." % (len(kernels), len(output_filters))

        concat = []
        for i in range(len(kernels)):
            x = Conv2D(output_filters[i], 1, use_bias=False)(inputs)
            x = Conv2DTranspose(output_filters[i], kernels[i], use_bias=False, strides=strides, padding=padding)(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = LeakyReLU(alpha=0.2)(x)

            # if kernels[i]>=3:
            #     x = Conv2D(output_filters[i], kernels[i]-2, use_bias=False)(x)
            concat += [x]

        x = Concatenate()(concat)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)
        return x


    def build_generator(self):

        noise = Input(shape=(self.latent_dim,))

        x = Reshape((1,1,self.latent_dim))(noise)

        # x = self.first_inception_block_up(x, 32, 1, [3,3,5,7])
        # x = self.inception_block_down(x, 32, 1, [3,3,3])

        # x = self.first_inception_block_up(x, 64, 2, [3,3,3,3])
        # x = self.inception_block_down(x, 64, 1, [3,3,3])

        # x = self.inception_block_up(x, 96, 2, [3,3,3])
        # x = self.inception_block_down(x, 96, 1, [3,3,3])

        # x = self.first_inception_block_up(x, 96, 1, [3,3,3])
        # x = self.inception_block_down(x, 96, 1, [3,3,3])

        # x = self.inception_block_up(x, 32, 2)
        # x = self.inception_block_down(x, 32, 1)

        x = Conv2DTranspose(self.latent_dim,4,use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # x = self.conv_inception_block_down(x, [96,48,24], [3,5,7], 1)
        # x = self.conv_inception_block_up(x, [96,48,24], [3,5,7], 2)

        x = self.conv_inception_block_down(x, [64,32,16], [3,5,7], 1)
        x = self.conv_inception_block_up(x, [64,32,16], [3,5,7], 2)

        x = self.conv_inception_block_down(x, [48,24,12], [3,5,7], 1)
        x = self.conv_inception_block_up(x, [48,24,12], [3,5,7], 2)

        x = self.conv_inception_block_down(x, [32,16,8], [3,5,7], 1)
        x = self.conv_inception_block_up(x, [32,16,8], [3,5,7], 2)

        x = Conv2D(self.channels,(3,3),padding='same')(x)
        img = Activation('linear')(x)
        
        model = Model(noise, img)

        model.summary()

        return Model(noise, img)

    def build_discriminator(self):

        img = Input(shape=self.img_shape)

        x = Conv2D(16,1,use_bias=False,padding='same')(img)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = self.conv_inception_block_down(x, [16,8,4], [3,5,7], 1)
        x = self.conv_inception_block_down(x, [32,16,8], [3,5,7], 2)

        x = self.conv_inception_block_down(x, [32,16,8], [3,5,7], 1)
        x = self.conv_inception_block_down(x, [48,24,12], [3,5,7], 2)

        x = self.conv_inception_block_down(x, [48,24,12], [3,5,7], 1)
        x = self.conv_inception_block_down(x, [64,32,16], [3,5,7], 2)

        x = Conv2D(self.latent_dim,4,use_bias=False)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = LeakyReLU(alpha=0.2)(x)

        # x = Conv2D(1024, kernel_size=2, strides=1)(x)
        # x = BatchNormalization(momentum=0.8)(x)
        # x = LeakyReLU(alpha=0.2)(x)
        # x = AveragePooling2D()(x)
        x = Flatten()(x)
        validity = Dense(1, activation='linear')(x)

        model = Model(img, validity)
        
        model.summary()

        return model

    def train(self, epochs, batch_size=128, save_interval=50, epoch_start=0):

        # Load the dataset
        (X_train, _), (_, _) = cifar10.load_data()

        # Rescale -1 to 1
        X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # print("Load data into memory...")
        # filenames = np.empty(0)
        # idx = 0
        # for root,_,files in os.walk(PATH):
        #     if len(files)>1:
        #         for i in range(len(files)):
        #             files[i] = root + '/' + files[i]
        #         filenames = np.append(filenames,files)

        # # max_size = len(filenames)
        # max_size = len(filenames)
        # X_train = np.empty((max_size,self.img_cols,self.img_rows,self.channels))
        # for i,f in tqdm(enumerate(filenames)):
        #     if i == max_size:
        #         break
        #     X_train[i] = sk.io.imread(f)/ 127. - 1.
        # print("done")

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epoch_start,epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.generator_model.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                # Plot the progress
                # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            
            if epoch>=2000 and epoch%(4*save_interval)==0:
                self.generator.save_weights(PATH_MODEL+'2019.05.22.doggan_10_cifar10_generator.'+str(epoch)+'.h5')
                self.critic.save_weights(PATH_MODEL+'2019.05.22.doggan_10_cifar10_discriminator.'+str(epoch)+'.h5')


        # for epoch in range(epoch_start,epochs):

        #     # ---------------------
        #     #  Train Discriminator
        #     # ---------------------

        #     # Select a random half of images
        #     idx = np.random.randint(0, X_train.shape[0], batch_size)
        #     imgs = X_train[idx]

        #     # Sample noise and generate a batch of new images
        #     noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
        #     gen_imgs = self.generator.predict(noise)

        #     # Train the discriminator (real classified as ones and generated as zeros)
        #     d_loss_real = self.discriminator.train_on_batch(imgs, valid)
        #     d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
        #     d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        #     # ---------------------
        #     #  Train Generator
        #     # ---------------------

        #     # Train the generator (wants discriminator to mistake images as real)
        #     g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            # print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))



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
        fig.savefig(PATH_SAVE+"cifar10_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=20000, batch_size=16, save_interval=200, epoch_start=0)
