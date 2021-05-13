import glob
import time

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import pathlib

from tensorflow.python.keras import Input
from tensorflow.python.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

from DoomData import DoomData


class DCGAN:
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = True

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = tf.keras.Sequential()

        model.add(layers.Dense(4 * 4 * 64, activation="relu", input_dim=self.latent_dim))
        model.add(layers.Reshape(target_shape=(4, 4, 64)))
        model.add(layers.Conv2DTranspose(
            filters=1024, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ))
        model.add(layers.Conv2DTranspose(
            filters=512, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ))

        model.add(layers.Conv2DTranspose(
            filters=256, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ))

        model.add(layers.Conv2DTranspose(
            filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ))

        model.add(layers.UpSampling2D())
        model.add(layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = tf.keras.Sequential()
        model.add(layers.InputLayer(input_shape=self.img_shape))
        model.add(layers.Conv2D(
            filters=128, kernel_size=3, strides=(2, 2), activation="relu"
        ))
        model.add(layers.Conv2D(
            filters=256, kernel_size=3, strides=(2, 2), activation="relu"
        ))
        model.add(layers.Conv2D(
            filters=512, kernel_size=3, strides=(2, 2), activation="relu"
        ))
        model.add(layers.Conv2D(
            filters=1024, kernel_size=3, strides=(2, 2), activation="relu"
        ))
        model.add(layers.Flatten())
        model.add(layers.Dense(units=1, activation="sigmoid"))


        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=28, save_interval=50, data_dir=""):

        # Load the dataset
        global noise, d_loss

        data = DoomData()
        x_train = data.get_trainable_data(data_dir)

        x_train = (x_train.astype(np.float32)) / 255.
        x_train = np.expand_dims(x_train, axis=3)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                imgs = x_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))
            #print("%d [W loss: %f]" % (epoch, self.wasserstein_loss()))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        predictions = self.generator.predict(noise)

        fig = plt.figure(figsize=(5, 5))
        for i in range(predictions.shape[0]):
            plt.subplot(5, 5, i + 1)
            plt.imshow(predictions[i, :, :, 0], cmap='gray')
            plt.axis('off')
        # 3 - Save the generated images
        fig.savefig("images/map_%d.png" % epoch)
        plt.close()
