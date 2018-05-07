from keras.utils.data_utils import get_file
import os
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm
from skimage.io import imread


def load_data():
    """Loads tiny-imagenet dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """

    dirname = 'tiny-imagenet-200'
    origin = 'cs231n.stanford.edu/tiny-imagenet-200.zip'
    path = get_file(dirname, origin=origin, untar=True, cache_dir='.')


    def load_train_images():
        subdir = 'train'
        X = np.empty((500 * 200, 64, 64, 3), dtype='uint8')
        Y = np.empty((500 * 200, ), dtype='int')
        classes = []
        for cls in os.listdir(os.path.join(path, subdir)):
            classes.append(cls)

        classes = {name: i for i, name in enumerate(classes)}
        i = 0
        for cls in tqdm(os.listdir(os.path.join(path, subdir))):
            for img in os.listdir(os.path.join(path, subdir, cls, 'images')):
                name = os.path.join(path, subdir, cls, 'images', img)
                X[i] = imread(name)
                Y[i] = classes[cls]
            i += 1

        return X, Y

    def load_test_images():
        X = np.empty((50 * 100, 64, 64, 3), dtype='unit8')
        Y = None
        i = 0
        for subdir in ('val', 'test'):
            for img in os.listdir(os.path.join(path, subdir, 'images')):
                name = os.path.join(path, subdir, 'images', img)
                X[i] = imread(name)
                i += 1
        return X, Y

    X_train, Y_train = load_train_images()
    X_test, Y_test = load_test_images()

    return (X_train, Y_train), X_test, Y_test
