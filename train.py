from DCGAN import DCGAN
from WGAN import WGAN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display
import pandas as pd

from DoomData import DoomData

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=9000, batch_size=1, save_interval=1, data_dir="map_images")
