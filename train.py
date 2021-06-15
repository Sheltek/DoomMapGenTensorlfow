from DCGAN import DCGAN

if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=40000, batch_size=1, save_interval=1, data_dir="map_images")
