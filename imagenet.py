import numpy as np
from gan.dataset import LabeledArrayDataset
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import shuffle
from tqdm import trange
from skimage.io import imread
from skimage.transform import resize
import os
from PIL import ImageFile
from skimage.color import gray2rgb

class ImageNetdataset(LabeledArrayDataset):
    def __init__(self, folder_train, batch_size=64, conditional=True,
                 noise_size=(128, ), image_size = (128, 128)):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.classes = os.listdir(folder_train)
        self.number_of_classes = len(self.classes)
        self.cache_dir = 'img_net_cache_' + str(image_size[0])
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
            if not os.path.isdir(os.path.join(self.folder_train, cls)):
                continue
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
            X = np.empty((end - image_index,) + self.image_size + (3,), dtype='uint8')
            Y = np.expand_dims(labels[image_index:end], axis=1)
            for i in trange(0, end - image_index):
                image = imread(names[image_index])
                if len(image.shape) == 2:
                   image = gray2rgb(image)
                if image.shape[2] == 4:
                    image = image[:,:,:3]
                X[i] = resize(image, self.image_size, preserve_range=True).astype('uint8')
                image_index += 1
            np.savez(bfile, x=X, y=Y)     

    def load_images_in_memmory(self, bucket_index):
        f = np.load(os.path.join(self.cache_dir, 'bucket_%s.npz' % bucket_index))
        self._X = f['x']
        self._Y = f['y']
        self._batches_before_shuffle = self._X.shape[0] // self._batch_size
        self._current_batch = 0

    def _load_discriminator_data(self, index):
        values = super(ImageNetdataset, self)._load_discriminator_data(index)
        values[0] = (values[0].astype('float32') - 127.5) / 127.5
        values[0] += np.random.uniform(0, 1/128.0, size=values[0].shape)
        return values

 
    def _shuffle_data(self):
        self.load_images_in_memmory(self.bucket_index)
        #print  len(os.listdir(self.cache_dir)) 
        self.bucket_index = (self.bucket_index + 1) % len(os.listdir(self.cache_dir))

    @property
    def _X_test(self):
        #number_of_images = 10000
	return self._X[:10000]
