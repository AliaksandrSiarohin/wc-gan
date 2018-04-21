from keras.utils.data_utils import get_file
import os
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte


def load_data():
    """Loads STL10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, None), (x_test, None)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """

    dirname = 'stl-10-python'
    origin = 'https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    path = get_file(dirname, origin=origin, untar=True, cache_dir='.')

    fpath = os.path.join(path, 'unlabeled_X.bin')

    X = np.fromfile(fpath, dtype=np.uint8)

    print X.shape
    X = np.reshape(X, (-1, 96, 96, 3))
    X_train = np.empty((X.shape[0], 48, 48, 3), dtype='uint8')

    for i in range(X.shape[0]):
        X_train[i] = img_as_ubyte(resize(X, (48, 48), preserve_range=True))

    np.random.seed(0)
    np.random.shuffle(X_train)

    return (X_train, None), (X_train[:10000], None)
