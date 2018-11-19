from keras.utils.data_utils import get_file
import os
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm
from skimage.io import imread
from skimage.color import gray2rgb
import pickle


def load_data():
    """Loads tiny-imagenet dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """

#    dirname = 'tiny-imagenet-200' 
    origin = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    path = get_file('tiny-imagenet-200.zip', origin=origin, extract=True, cache_dir='.', archive_format='zip')
    path = path.replace('.zip', '')

    def load_train_images():
        subdir = 'train'
        X = np.empty((500 * 200, 64, 64, 3), dtype='uint8')
        Y = np.empty((500 * 200, ), dtype='int')
        classes = []
        for cls in os.listdir(os.path.join(path, subdir)):
            classes.append(cls)
#        f = open('ti_classses.pkl', 'w')
#        pickle.dump(classes, f)
#        f.close()
        classes = {name: i for i, name in enumerate(classes)}     
        i = 0
        for cls in tqdm(os.listdir(os.path.join(path, subdir))):
            for img in os.listdir(os.path.join(path, subdir, cls, 'images')):
                name = os.path.join(path, subdir, cls, 'images', img)
                image = imread(name)
                if len(image.shape) == 2:
                    image = gray2rgb(image)
                X[i] = image
                Y[i] = classes[cls]
                i += 1
        print i
        return X, Y

    def load_test_images():
        X = np.empty((100 * (50 + 50), 64, 64, 3), dtype='uint8')
        Y = None
        i = 0
        for subdir in ('test', ):
            for img in tqdm(os.listdir(os.path.join(path, subdir, 'images'))):
                name = os.path.join(path, subdir, 'images', img)
                image = imread(name)
                if len(image.shape) == 2:
                    image = gray2rgb(image)
                X[i] = image
                i += 1
        print i
        return X, Y
    print ("Loading images...")
    X_train, Y_train = load_train_images()
    X_test, Y_test = load_test_images()

    return (X_train, Y_train), (X_test, Y_test)
