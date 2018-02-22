from keras.models import Model, Input
from keras.layers import Conv2D, LeakyReLU, Activation, Reshape, UpSampling2D, Dense, Flatten, AveragePooling2D, GlobalAveragePooling2D, Lambda
from keras.backend import tf as ktf
from keras.engine.topology import Layer
import keras.backend as K
from keras.layers.merge import Add
from keras import initializers

import numpy as np

from cmd import parser_with_default_args
from dataset import UGANDataset
from wgan_gp import WGAN_GP
from gan_gp import GAN_GP
from train import Trainer
from functools import partial


iter_count = K.variable(0, name='iter_count', dtype='int32')
alpha = K.variable(0, name='alpha', dtype='float32')

block_filter_size = [512, 512, 512, 512, 256, 128, 64, 32, 16]

class ProgresiveGrowingG(Layer):
    def __init__(self, n_iters_per_stage, final_size, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros', **kwargs):
        self.n_iters_per_stage = n_iters_per_stage
        self.final_size = final_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        super(ProgresiveGrowingG, self).__init__(**kwargs)

    def build(self, input_shape):
        self.noise_size = input_shape[1]
        self.resize_fn = lambda x: K.resize_images(x, 2, 2, "channels_last")
        exp_number_of_blocks = min(self.final_size) / 4
        ch = self.noise_size / (self.final_size[0] * self.final_size[1] / (exp_number_of_blocks **2))

        self.input_resized_shape = (-1, self.final_size[0] / exp_number_of_blocks,
                                    self.final_size[1] / exp_number_of_blocks, ch)

        number_of_blocks = exp_number_of_blocks.bit_length()
        self.number_of_blocks = number_of_blocks

        self.first_conv_params = []
        self.second_conv_params = []
        self.to_rgb_conv_params = []

        for i in range(number_of_blocks):
            kernel_shape = [4, 4] if i == 0 else [3, 3]
            kernel_shape += [ch, block_filter_size[i]]

            kernel = self.add_weight("block%s_conv0_kernel" % i, tuple(kernel_shape),
                                      initializer=self.kernel_initializer)
            bias = self.add_weight("block%s_conv0_bias" % i, (block_filter_size[i],),
                                      initializer=self.kernel_initializer)

            self.first_conv_params.append((kernel, bias))
            ch = block_filter_size[i]

        for i in range(number_of_blocks):
            kernel = self.add_weight("block%s_conv1_kernel" % i, [3, 3, block_filter_size[i], block_filter_size[i]],
                                      initializer=self.kernel_initializer)
            bias = self.add_weight("block%s_conv1_bias" % i, (block_filter_size[i],),
                                      initializer=self.kernel_initializer)

            self.second_conv_params.append((kernel, bias))

        for i in range(number_of_blocks):
            kernel = self.add_weight("block%s_torgb_kernel" % i, [1, 1, block_filter_size[i], 3],
                                      initializer=self.kernel_initializer)
            bias = self.add_weight("block%s_torgb_bias" % i, (3,),
                                      initializer=self.kernel_initializer)

            self.to_rgb_conv_params.append((kernel, bias))

        super(ProgresiveGrowingG, self).build(input_shape)

    def call(self, inputs, **kwargs):
        stage_number = iter_count / self.n_iters_per_stage

        def apply_conv(out, params, activation):
            assert activation in ['relu', 'tanh']
            out = K.conv2d(out, params[0], padding='same')
            out = K.bias_add(out, params[1])
            if activation == 'relu':
                out = K.relu(out, alpha=0.2)
            else:
                out = K.tanh(out)
            return out

        def output_for_stage(stage_number_int):
            out = inputs
            out = K.reshape(out, self.input_resized_shape)
            for block_index in range(stage_number_int / 2 + 1):
                if block_index != 0:
                    out = self.resize_fn(out)
                out = apply_conv(out, self.first_conv_params[block_index], 'relu')
                out = apply_conv(out, self.second_conv_params[block_index], 'relu')
            torgb1 = apply_conv(out, self.to_rgb_conv_params[stage_number_int / 2], 'tanh')

            if stage_number_int % 2 != 0:
                out = self.resize_fn(out)
                torgb1 = self.resize_fn(torgb1)
                out = apply_conv(out, self.first_conv_params[stage_number_int / 2 + 1], 'relu')
                out = apply_conv(out, self.second_conv_params[stage_number_int / 2 + 1], 'relu')
                torgb2 = apply_conv(out, self.to_rgb_conv_params[stage_number_int / 2 + 1], 'tanh')

                return (1 - alpha) * torgb1 + alpha * torgb2
            else:
                return torgb1

        pairs = []
        for i in range(2 * self.number_of_blocks - 1):
            pairs.append( (ktf.equal(stage_number, i), partial(output_for_stage, stage_number_int=i)))

        return ktf.case(pairs, default=lambda: output_for_stage(2 * self.number_of_blocks - 2))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], None, None, 3)


    def get_config(self):
        config = {'n_iters_per_stage': self.n_iters_per_stage,
                  'final_size': self.final_size}
        base_config = super(ProgresiveGrowingG, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ProgresiveGrowingD(Layer):
    def __init__(self, n_iters_per_stage, final_size, kernel_initializer='glorot_uniform',
                                bias_initializer='zeros', **kwargs):
        self.n_iters_per_stage = n_iters_per_stage
        self.final_size = final_size
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        super(ProgresiveGrowingD, self).__init__(**kwargs)

    def build(self, input_shape):
        self.resize_fn = lambda x: K.pool2d(x, (2, 2), strides=(2, 2),
                                        data_format="channels_last", pool_mode='avg', padding='valid')
        exp_number_of_blocks = min(self.final_size) / 4

        number_of_blocks = exp_number_of_blocks.bit_length()
        self.number_of_blocks = number_of_blocks

        self.final_filter_size = [self.final_size[0] / exp_number_of_blocks,
                                    self.final_size[1] / exp_number_of_blocks]

        self.first_conv_params = []
        self.second_conv_params = []
        self.from_rgb_conv_params = []

        for i in range(number_of_blocks):
            ch_from = block_filter_size[i + 1]
            ch_to = block_filter_size[i]
            kernel = self.add_weight("block%s_conv0_kernel" % i, [3, 3, ch_from, ch_to],
                                      initializer=self.kernel_initializer)
            bias = self.add_weight("block%s_conv0_bias" % i, (block_filter_size[i],),
                                      initializer=self.kernel_initializer)

            self.first_conv_params.append((kernel, bias))


        for i in range(number_of_blocks):
            kernel_shape = self.final_filter_size if i == 0 else [3, 3]
            kernel_shape += [block_filter_size[i], block_filter_size[i]]

            kernel = self.add_weight("block%s_conv1_kernel" % i, tuple(kernel_shape),
                                      initializer=self.kernel_initializer)
            bias = self.add_weight("block%s_conv1_bias" % i, (block_filter_size[i],),
                                      initializer=self.kernel_initializer)

            self.second_conv_params.append((kernel, bias))

        for i in range(number_of_blocks):
            kernel = self.add_weight("block%s_fromrgb_kernel" % i, [1, 1, 3, block_filter_size[i + 1]],
                                      initializer=self.kernel_initializer)
            bias = self.add_weight("block%s_fromrgb_bias" % i, (block_filter_size[i + 1],),
                                      initializer=self.kernel_initializer)

            self.from_rgb_conv_params.append((kernel, bias))

        self.dense = self.add_weight('fc', (block_filter_size[0], 1), initializer=self.kernel_initializer)

        super(ProgresiveGrowingD, self).build(input_shape)

    def call(self, inputs, **kwargs):
        stage_number = iter_count / self.n_iters_per_stage

        def apply_conv(out, params, pad='same'):
            out = K.conv2d(out, params[0], padding=pad)
            out = K.bias_add(out, params[1])
            out = K.relu(out, alpha=0.2)
            return out

        def output_for_stage(stage_number_int):
            out = inputs

            last_block_index = (stage_number_int + 1)/2

            out = apply_conv(out, self.from_rgb_conv_params[last_block_index])
            out = apply_conv(out, self.first_conv_params[last_block_index])

            if last_block_index != 0:
                out = apply_conv(out, self.second_conv_params[last_block_index])
                out = self.resize_fn(out)
            else:
                out = apply_conv(out, self.second_conv_params[last_block_index], pad='valid')

            if stage_number_int % 2 != 0:
                fromrgb = self.resize_fn(inputs)
                fromrgb = apply_conv(fromrgb, self.from_rgb_conv_params[last_block_index - 1])
                out = (1 - alpha) * fromrgb + alpha * out

            for block_index in range(last_block_index - 1, -1, -1):
                out = apply_conv(out, self.first_conv_params[block_index])
                if block_index != 0:
                    out = apply_conv(out, self.second_conv_params[block_index])
                    out = self.resize_fn(out)
                else:
                    out = apply_conv(out, self.second_conv_params[block_index], 'valid')

            out = K.reshape(out, (-1, block_filter_size[0]))
            return K.dot(out, self.dense)

        pairs = []
        for i in range(2 * self.number_of_blocks - 1):
            pairs.append( (ktf.equal(stage_number, i), partial(output_for_stage, stage_number_int=i)))

        return ktf.case(pairs, default=lambda: output_for_stage(2 * self.number_of_blocks - 2))


    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


    def get_config(self):
        config = {'n_iters_per_stage': self.n_iters_per_stage,
                  'final_size': self.final_size}
        base_config = super(ProgresiveGrowingD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



def make_generator(noise_size, final_size, n_iters_per_stage):
    inp = Input((noise_size, ))
    out = ProgresiveGrowingG(n_iters_per_stage, final_size)(inp)
    # out = Reshape((8, 4, 16)) (inp)
    # out = Conv2D(512, (4, 4), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    # out = Conv2D(512, (3, 3), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    # torgb1 = Conv2D(3, (1, 1,)) (out)
    # torgb1 = UpSampling2D()(torgb1)
    # torgb1 = Activation('tanh') (torgb1)
    #
    # out = UpSampling2D()(out)
    # out = Conv2D(512, (3, 3), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    # out = Conv2D(512, (3, 3), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    # torgb2 = Conv2D(3, (1, 1,)) (out)
    # torgb2 = Activation('tanh') (torgb2)
    #
    # out = Lambda(lambda inp: (1 - alpha) * inp[0] + alpha * inp[1]) ([torgb1, torgb2])
    # out = Lambda(lambda x: x, output_shape=(None, None, 3))(out)
    return Model(inp, out)

def make_discriminator(gan_type, final_size, n_iters_per_stage):
    inp = Input((None, None, 3))

    # out = Conv2D(512, (1, 1)) (inp)
    # out = LeakyReLU(0.2)(out)
    # out = Conv2D(512, (3, 3), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    # out = Conv2D(512, (3, 3), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    # out = AveragePooling2D()(out)
    #
    # # from_rgb1 = AveragePooling2D()(inp)
    # # from_rgb1 = Conv2D(512, (1, 1), padding='same')(from_rgb1)
    # # from_rgb1 = LeakyReLU(0.2)(from_rgb1)
    # #
    # # out = Lambda(lambda inp: (1 - alpha) * inp[0] + alpha * inp[1]) ([from_rgb1, out])
    #
    # out = Conv2D(512, (3, 3), padding='same')(out)
    # out = LeakyReLU(0.2)(out)
    #
    # out = Conv2D(512, (8, 4), padding='valid')(out)
    # out = LeakyReLU(0.2)(out)
    #
    # out = Lambda(lambda x: K.reshape(x, (-1, 512)), output_shape=(512, ))(out)
    # out = Dense(1, use_bias=False) (out)
    out = ProgresiveGrowingD(n_iters_per_stage, final_size)(inp)
    if gan_type == 'gan':
        out = Activation('sigmoid')(out)
    return Model(inp, out)


import os
from skimage.transform import resize
from skimage import img_as_ubyte
from skimage.io import imread

class FolderDataset(UGANDataset):
    def __init__(self, input_dir, batch_size, noise_size, image_size, iters_per_stage):
        super(FolderDataset, self).__init__(batch_size, noise_size)
        self._image_names = np.array(os.listdir(input_dir))
        self._input_dir = input_dir
        self._image_size = image_size
        self._iters_per_stage = iters_per_stage
        self._batches_before_shuffle = int(self._image_names.shape[0] // self._batch_size)
        self._iter_count = 0

    def number_of_batches_per_epoch(self):
        return 1000

    def _preprocess_image(self, img):
        stage_number = self._iter_count / self._iters_per_stage
        resolution = (stage_number + 1) / 2
        blocks = min(self._image_size).bit_length() - 3
        image_size = (self._image_size[0] / (2 ** (blocks - resolution)),
                      self._image_size[1] / (2 ** (blocks - resolution)))
        image_size = min(self._image_size[0], image_size[0]), min(self._image_size[1], image_size[1])
        return resize(img, image_size) * 2 - 1

    def _deprocess_image(self, img):
        return img_as_ubyte((img + 1) / 2)

    def _load_discriminator_data(self, index):
        self._iter_count += self._batch_size
        alpha_val = self._iter_count / float(self._iters_per_stage) - (self._iter_count / self._iters_per_stage)
        K.set_value(alpha, alpha_val)
        K.set_value(iter_count, self._iter_count)
        data = [np.array([self._preprocess_image(imread(os.path.join(self._input_dir, img_name)))
                          for img_name in self._image_names[index]])]
        return data

    def _shuffle_data(self):
        np.random.shuffle(self._image_names)

    def display(self, output_batch, input_batch = None):
        image = super(FolderDataset, self).display(output_batch)
        return self._deprocess_image(image)

def main():
    parser = parser_with_default_args()
    parser.add_argument("--input_dir", default='../data/market-dataset/bounding_box_train',
                        help='Foldet with input images')
    parser.add_argument("--gan_type", choices =['gan', 'wgan'], default='wgan', help='Type of gan to use')
    parser.add_argument("--iters_per_stage", type=int, default=int(1e5), help="Number of iters in each stage paper (6e5)")


    args = parser.parse_args()
    n_iters_per_stage = args.iters_per_stage
    args.batch_size = 16
    args.training_ratio = 1

    image_size = (128, 64)
    generator = make_generator(512, image_size, n_iters_per_stage=n_iters_per_stage)

    discriminator = make_discriminator(args.gan_type, image_size, n_iters_per_stage=n_iters_per_stage)
    # from keras.optimizers import SGD
    # discriminator.compile(loss='mse', optimizer=SGD())

    dataset = FolderDataset(args.input_dir, args.batch_size, (512, ), image_size, iters_per_stage = n_iters_per_stage)
    gan_type = GAN_GP if args.gan_type == 'gan' else WGAN_GP
    gan = gan_type(generator, discriminator, **vars(args))
    trainer = Trainer(dataset, gan, **vars(args))

    trainer.train()

if __name__ == "__main__":
    main()



