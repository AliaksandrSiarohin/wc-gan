import numpy as np
from gan.dataset import LabeledArrayDataset
from keras.preprocessing.image import ImageDataGenerator


class ImageNetdataset(LabeledArrayDataset):
    def __init__(self, folder_train, image_size=(64, 64), batch_size=64, conditional=True,
                 noise_size=(128, )):
        #dequantize
        rescale = lambda x: (x - 127.5) / 127.5 + np.random.uniform(0, 1/128.0, size=x.shape)
        train_datagen = ImageDataGenerator(preprocessing_function=rescale)

        self.train_flow = train_datagen.flow_from_directory(folder_train, target_size=image_size, batch_size=batch_size)

        self.number_of_classes = len(self.train_flow.classes)
        self.conditional = conditional
        self.image_size = image_size

        self._noise_size = noise_size
        self._batch_size = batch_size
        self._batches_before_shuffle = 10000000

    def number_of_batches_per_epoch(self):
        return 1000

    def number_of_batches_per_validation(self):
        return 0

    def next_generator_sample(self):
        labels = [] if not self.conditional is None else self.current_discriminator_labels
        return [np.random.normal(size=(self._batch_size,) + self._noise_size)] + labels

    def next_generator_sample_test(self):
        labels = [] if not self.conditional is None else [np.random.randint(self.number_of_classes, size=(self._batch_size, 1))]
        return [np.random.normal(size=(self._batch_size,) + self._noise_size)] + labels

    def _load_discriminator_data(self, index):
        X, self.current_discriminator_labels = self.train_flow.next()
        if not self.conditional:
            self.current_discriminator_labels = []
        return [self._X[index]] + self.current_discriminator_labels

    def _shuffle_data(self):
        None #shuling done in datagenerator

    @property
    def X_test(self):
        number_of_images = 10000
        images = np.empty((number_of_images, ) + self.image_size + (3, ))

        for begin in range(0, number_of_images, self._batch_size):
            end = min(number_of_images, begin + self._batch_size)
            n_images = end - begin
            x, _ = self.train_flow.next()        
            images[begin:end] = x

        return images
