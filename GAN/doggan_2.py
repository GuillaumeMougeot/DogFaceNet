from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, Lambda
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt

import numpy as np

import os
import skimage as sk


PATH = '../data/dogfacenet/aligned/after_4_bis/'
PATH_SAVE = '../output/history/'
PATH_MODEL = '../output/model/'
VALID_SPLIT = 0.1
TEST_SPLIT = 0.1


class AdversarialAutoencoder():
    def __init__(self):
        self.img_rows = 224
        self.img_cols = 224
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        img = Input(shape=self.img_shape)

        h = Conv2D(16,(3,3),strides=(2,2))(img)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization()(h)

        for layer in [32,32,64,64,128]:
            h = Conv2D(layer,(3,3),strides=(2,2))(h)
            h = LeakyReLU(alpha=0.2)(h)
            h = BatchNormalization()(h)

        h = Conv2D(256,(2,2))(h)
        h = LeakyReLU(alpha=0.2)(h)

        h = Flatten()(h)

        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = Lambda(lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                output_shape=lambda p: p[0])([mu, log_var])

        model = Model(img, latent_repr)
        print("Encoder model: ")
        model.summary()

        return model

    def build_decoder(self):

        z = Input(shape=(self.latent_dim,))

        h = Reshape((1,1,self.latent_dim))(z)
        h = Conv2DTranspose(128,(2,2))(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization()(h)

        filters = [64,64,32,32,16]
        kernels = [4,3,3,3,3]

        for i in range(len(filters)):
            h = Conv2DTranspose(filters[i],kernels[i],strides=(2,2))(h)
            h = LeakyReLU(alpha=0.2)(h)
            h = BatchNormalization()(h)

        h = Conv2DTranspose(self.channels,(4,4),strides=(2,2))(h)
        img = LeakyReLU(alpha=0.2)(h)

        model = Model(z, img)
        print("Decoder model: ")
        model.summary()

        return model

    def build_discriminator(self):

        z = Input(shape=(self.latent_dim,))

        h = Reshape((1,1,self.latent_dim))(z)

        # Up
        h = Conv2DTranspose(32,(3,3))(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = BatchNormalization()(h)

        for layer in [64,64,128,256]:
            h = Conv2DTranspose(layer,(3,3),strides=(2,2))(h)
            h = LeakyReLU(alpha=0.2)(h)
            h = BatchNormalization()(h)

        # Down
        for layer in [128,64,64,32]:
            h = Conv2D(64,(3,3),strides=(2,2))(h)
            h = LeakyReLU(alpha=0.2)(h)
            h = BatchNormalization()(h)

        h = Conv2D(1,(3,3),activation='sigmoid')(h)
        h = Flatten()(h)
        model = Model(z, h)
        print("Discriminator model: ")
        model.summary()

        return model

    def train(self, epochs, batch_size=128, sample_interval=50):

        # # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()

        # # Rescale -1 to 1
        # X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # X_train = np.expand_dims(X_train, axis=3)

        # Load datas
        print("Load data into memory...")
        filenames = np.empty(0)
        labels = np.empty(0)
        idx = 0
        for root,_,files in os.walk(PATH):
            if len(files)>1:
                for i in range(len(files)):
                    files[i] = root + '/' + files[i]
                filenames = np.append(filenames,files)
                labels = np.append(labels,np.ones(len(files))*idx)
                idx += 1
        print(len(labels))

        nbof_classes = len(np.unique(labels))
        print(nbof_classes)

        max_size = len(filenames)
        X_train = np.empty((max_size,self.img_cols,self.img_rows,self.channels))
        for i,f in enumerate(filenames):
            if i == max_size:
                break
            X_train[i] = sk.io.imread(f)/ 255.0

        
        print("done")


        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.adversarial_autoencoder.save(PATH_MODEL+"doggan."+str(epoch)+".h5")

    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images_dogs/dogs_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        #save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")


if __name__ == '__main__':
    aae = AdversarialAutoencoder()
    aae.train(epochs=20001, batch_size=32, sample_interval=200)


