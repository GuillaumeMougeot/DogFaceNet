from __future__ import print_function, division

from keras.datasets import cifar10
# from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import RMSprop

import keras.backend as K

import matplotlib.pyplot as plt
import skimage as sk
import sys

import numpy as np

PATH_SAVE = '../output/images_AWS/'
PATH_MODEL = '../output/model/gan/wgan/'

class WGAN():
    def __init__(self):
        self.depth = 0
        self.img_rows = 2**(self.depth+2)
        self.img_cols = 2**(self.depth+2)
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 128
        

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.02
        self.optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

    def rebuild(self):
        # Increase depth of the network
        assert self.depth+1<=3, "Too deep!"
        self.depth += 1

        # Change input size
        self.img_rows = 2**(self.depth+2)
        self.img_cols = 2**(self.depth+2)
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)   

        # self.clip_value = self.clip_value/2

        # Build and compile the critic
        self.critic = self.add_layer_critic(self.critic)
        self.critic.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.add_layer_generator(self.generator)

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
            optimizer=self.optimizer,
            metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()
        # model.add(Input(shape=(self.latent_dim,)))
        model.add(Reshape((1, 1, self.latent_dim), input_shape=(self.latent_dim,)))

        model.add(UpSampling2D((4,4)))

        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))


        filters = 64
        for _ in range(self.depth):
            model.add(UpSampling2D())

            model.add(Conv2D(filters, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            model.add(Conv2D(filters, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(Activation("relu"))
            filters = filters // 2

        model.add(Conv2D(self.channels, kernel_size=1, padding="same"))
        model.add(Activation("linear"))

        # noise = Input(shape=(self.latent_dim,))
        # img = model(noise)

        model.summary()

        return model

    def add_layer_generator(self, base_model):
        model = Sequential()
        for l in base_model.layers[:-2]:
            model.add(l)

        filters = 2**(6-self.depth)

        model.add(UpSampling2D())
        for _ in range(2):
            model.add(Conv2D(filters, kernel_size=3, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(self.channels, kernel_size=1, padding="same"))
        model.add(Activation("linear"))

        model.summary()

        return model



    def build_critic(self):

        model = Sequential()

        filters = 2**(7-self.depth)

        model.add(Conv2D(filters, kernel_size=1, strides=1, padding="same", input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        for _ in range(self.depth):
            model.add(Conv2D(filters, kernel_size=3, strides=2, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            filters *= 2
            model.add(Conv2D(filters, kernel_size=3, strides=1, padding="same"))
            model.add(BatchNormalization(momentum=0.8))
            model.add(LeakyReLU(alpha=0.2))
            model.add(Dropout(0.25))

        model.add(Conv2D(128, kernel_size=4, strides=1))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        
        model.summary()
        return model

    def add_layer_critic(self, base_model):
        filters = 2**(7-self.depth)
        
        model = Sequential()
        model.add(Conv2D(filters, kernel_size=1, strides=1, padding="same", input_shape=self.img_shape))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(filters, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        filters *= 2
        model.add(Conv2D(filters, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.summary()

        for l in base_model.layers[3:]:
            model.add(l)
        
        model.summary()

        return model

    def resize_data(self, data, ratio=2):
        """
        Resize the images in data by a factor of ratio.
        """
        l,h,w,c = data.shape
        shape_out = (h//ratio,w//ratio,c)
        data_out = np.empty((l,h//ratio,w//ratio,c))
        print("Resizing dataset...")
        for i in range(len(data_out)):
        # for i in range(4):
            data_out[i] = sk.transform.resize(data[i],shape_out)
        print("Done!")
        return data_out

    def train_unit(self, X_train, epochs, batch_size=128, sample_interval=50, start_epoch=0):
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(start_epoch, epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                
                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                clip_ratio = self.clip_value
                for l in self.critic.layers:
                    weights = l.get_weights()
                    if len(weights)>0:
                        weights = [np.clip(w, -clip_ratio, clip_ratio) for w in weights]
                        l.set_weights(weights)
                        # clip_ratio /= 1.5


            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
            if epoch > 2000 and epoch % (sample_interval*4) == 0:
                self.generator.save_weights(PATH_MODEL+'wgan_prog_cifar10.gen.'+str(epoch)+'.h5')
                self.critic.save_weights(PATH_MODEL+'wgan_prog_cifar10.cri.'+str(epoch)+'.h5')



    def train(self,
            epochs,
            batch_size=128,
            sample_interval=50,
            model_update=2000,
            start_epoch=0
            ):

        # Load the dataset
        (X_train, _), (_, _) = cifar10.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        ratio = 2**(3-self.depth)
        start_model_update=0
        for _ in range(int(np.log2(ratio))):
            data_resized = self.resize_data(X_train, ratio)
            self.train_unit(
                data_resized,
                model_update+start_model_update,
                batch_size=batch_size,
                sample_interval=sample_interval,
                start_epoch=start_model_update)
            start_model_update += model_update
            ratio = ratio // 2
            self.rebuild()

        self.train_unit(
            X_train,
            epochs - model_update+start_model_update,
            batch_size=batch_size,
            sample_interval=sample_interval,
            start_epoch=start_model_update)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("../output/images/wgan/cifar10/cifar10_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    wgan = WGAN()
    wgan.train(epochs=80000, batch_size=64, sample_interval=50, model_update=10000)
