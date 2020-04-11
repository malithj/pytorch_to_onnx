import numpy as np
from functools import partial
import pickle


def main():
    img_size = 224
    n_batches = 1
    n_channels = 3
    file_name = 'resources/validation_imagenet.npy'
    data = np.random.rand(n_batches * img_size * img_size * n_channels)
    data = data.astype(np.float32)
    print("Writing data array to file: {0:}".format(file_name))
    data.tofile(file_name)


if __name__ == '__main__':
    main()
