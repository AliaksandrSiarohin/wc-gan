from keras.utils.data_utils import get_file
import os
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm


def load_data():
    """Loads STL10 dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, None), (x_test, None)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """

    dirname = 'stl10_binary'
    origin = 'https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
    path = get_file(dirname, origin=origin, untar=True, cache_dir='.')

    fpath = os.path.join(path, 'unlabeled_X.bin')

    X = np.fromfile(fpath, dtype=np.uint8)

    print X.shape
    X = np.reshape(X, (-1, 3, 96, 96))
    X = np.transpose(X, (0, 3, 2, 1))
    X_train = np.empty((X.shape[0], 48, 48, 3), dtype='uint8')

    print ("Resising images...")
    for i in tqdm(range(X.shape[0])):
        X_train[i] = img_as_ubyte(resize(X[i], (48, 48)))

    np.random.seed(0)
    np.random.shuffle(X_train)

    return (X_train, None), (X_train[:10000], None)
