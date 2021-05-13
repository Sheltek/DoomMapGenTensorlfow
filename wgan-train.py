import numpy as np
import pandas as pd

from DoomData import DoomData
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display

from WGAN import WGAN

data_dir = "map_images"
data = DoomData()
train_dataset = data.get_trainable_data(data_dir)

img_rows = 128
img_cols = 128
channels = 1
img_shape = (img_rows, img_cols, channels)

N_Z = 64
TRAIN_BUF = 60000
BATCH_SIZE = 32
TEST_BUF = 10000
N_TRAIN_BATCHES = int(TRAIN_BUF / BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF / BATCH_SIZE)

generator = [
    layers.Dense(4 * 4 * 64, activation="relu"),
    layers.Reshape(target_shape=(4, 4, 64)),
    layers.Conv2DTranspose(
        filters=1024, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    layers.Conv2DTranspose(
        filters=512, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    layers.Conv2DTranspose(
        filters=256, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    layers.Conv2DTranspose(
        filters=128, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
    ),
    layers.UpSampling2D(),
    layers.Conv2DTranspose(
        filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
    )
]

discriminator = [
    layers.InputLayer(input_shape=img_shape),
    layers.Conv2D(
        filters=128, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    layers.Conv2D(
        filters=256, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    layers.Conv2D(
        filters=512, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    layers.Conv2D(
        filters=1024, kernel_size=3, strides=(2, 2), activation="relu"
    ),
    layers.Flatten(),
    layers.Dense(units=1, activation="sigmoid")
]

# optimizers
gen_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)
disc_optimizer = tf.keras.optimizers.RMSprop(0.0005)  # train the model
# model
model = WGAN(
    gen=generator,
    disc=discriminator,
    gen_optimizer=gen_optimizer,
    disc_optimizer=disc_optimizer,
    n_Z=N_Z,
    gradient_penalty_weight=10.0
)


# exampled data for plotting results
def plot_reconstruction(model, nex=8, zm=2):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    predictions = model.generate(tf.random.normal(shape=(0, 1, (r * c, 100))))

    fig = plt.figure(figsize=(5, 5))
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    # 3 - Save the generated images
    fig.savefig("images/map_%d.png" % epoch)
    plt.close()


losses = pd.DataFrame(columns=['disc_loss', 'gen_loss'])

n_epochs = 200
for epoch in range(n_epochs):
    loss = []
    # train
    for batch, train_x in tqdm(
        zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
    ):
        model.train(train_x)
        loss.append(model.compute_loss(train_x))

    losses.loc[len(losses)] = np.mean(loss, axis=0)
    # plot results
    display.clear_output()
    print(
        "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
            epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
        )
    )
    plot_reconstruction(model)
