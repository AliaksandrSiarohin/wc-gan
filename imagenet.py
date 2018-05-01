import numpy as np
from gan.dataset import LabeledArrayDataset
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from tqdm import trange
from skimage.io import imread
import os
from PIL import ImageFile

class ImageNetdataset(LabeledArrayDataset):
    def __init__(self, folder_train, batch_size=64, conditional=True,
                 noise_size=(128, ), image_size = (64, 64)):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.classes = os.listdir(folder_train)
        self.number_of_classes = len(self.classes)
        self.cache_dir = 'img_net_cache'
        self.conditional = conditional
        self.folder_train = folder_train
        
        self._noise_size = noise_size
        self._batch_size = batch_size
        self.image_size = image_size
        self.split_in_buckets()
        self.load_images_in_memmory(0)
        self.bucket_index = 0
     
    def split_in_buckets(self, bucket_size = 100000):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        image_names = []
        image_classes = []
        for i, cls in enumerate(self.classes):
            for img in os.listdir(os.path.join(self.folder_train, cls)):
                image_names.append(os.path.join(self.folder_train, cls, img))
                image_classes.append(i)
        names, labels = shuffle(image_names, image_classes, random_state = 0)
        print ("Number of images %s, of classes %s" % (len(names), len(np.unique(labels))))
        print ("Preprocessing images...")
        image_index = 0
        for bucket_index in range(len(names) / bucket_size  + 1):
            bfile = os.path.join(self.cache_dir, 'bucket_%s.npz' % bucket_index)
            if os.path.exists(bfile):
               image_index += bucket_size
               continue                
            end = min(image_index + bucket_size, len(names))
            X = np.empty((end - image_index,) + self.image_size + (3,), dtype='float32')
            Y = np.expand_dims(labels[image_index:end], axis=1)
            for i in trange(0, end - image_index):
                X[i] = (imread(names[image_index]) - 127.5) / 127.5
                image_index += 1
            np.savez(bfile, x=X, y=Y)     
          
    def load_images_in_memmory(self, bucket_index):
        f = np.load(os.path.join(self.cache_dir, 'bucket_%s.npz' % bucket_index))
        self._X = f['x']
        self._Y = f['y']
        self._batches_before_shuffle = self._X.shape[0] // self._batch_size
        self._current_batch = 0

    def _shuffle_data(self):
        self.load_images_in_memmory(self.bucket_index)
        #print  len(os.listdir(self.cache_dir)) 
        self.bucket_index = (self.bucket_index + 1) % len(os.listdir(self.cache_dir))

    @property
    def _X_test(self):
        #number_of_images = 10000
	return self._X[:10000]
