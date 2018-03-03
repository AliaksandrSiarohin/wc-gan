from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, Layer
from keras.layers import BatchNormalization, Add, Embedding, Concatenate, UpSampling2D, AveragePooling2D, Subtract


import numpy as np
from gan.layer_utils import glorot_init

from gan.conditional_layers import ConditinalBatchNormalization, cond_resblock, ConditionalConv11

import keras.backend as K


class Mul(Layer):
    ITER_COUNT = K.variable(0, dtype='int32', name='alpha')

    def __init__(self, number_of_iters_per_stage=10000, update_count=True, **kwargs):
        super(Mul, self).__init__(**kwargs)
        self.number_of_iters_per_stage = number_of_iters_per_stage
        self.update_count = update_count

    def call(self, inputs, **kwargs):
        if self.update_count:
            self.add_update(K.update_add(Mul.ITER_COUNT, 1))
        return (1.0 - K.cast(Mul.ITER_COUNT, dtype='float32') / self.number_of_iters_per_stage) * inputs

    @staticmethod
    def reset():
        K.set_value(Mul.ITER_COUNT, 0)


def make_generator(input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ), block_sizes=(128, 128, 128),
                   first_block_shape=(4, 4, 128), number_of_classes=10, concat_cls=False,
                   conditional_bottleneck=False, unconditional_bottleneck=False,
                   conditional_shortcut=False, unconditional_shortcut=True,
                   conditional_bn=False, progressive=False, progressive_stage=0, progressive_iters_per_stage=10000):

    assert conditional_shortcut or unconditional_shortcut


    inp = Input(input_noise_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if concat_cls:
        y = Embedding(input_dim=number_of_classes, output_dim=block_sizes[0])(cls)
        y = Concatenate(axis=-1)([y, inp])
    else:
        y = inp

    y = Dense(np.prod(first_block_shape), kernel_initializer=glorot_init)(y)
    y = Reshape(first_block_shape)(y)

    if conditional_bn:
        norm = lambda axis, name: (lambda inp: ConditinalBatchNormalization(number_of_classes=number_of_classes,
                                                                                axis=axis, name=name)([inp, cls]))
    else:
        norm = BatchNormalization


    if not progressive:
        for i, block_size in enumerate(block_sizes):
            y = cond_resblock(y, cls, kernel_size=(3, 3), resample="UP", nfilters=block_size, number_of_classes=number_of_classes,
                          name='Generator.' + str(i), norm=norm, is_first=False, conv_layer=Conv2D, cond_conv_layer=ConditionalConv11,
                          cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                          cond_shortcut=conditional_shortcut, uncond_shortcut=unconditional_shortcut)

        y = BatchNormalization(axis=-1)(y)
        y = Activation('relu')(y)
        output = Conv2D(output_channels, (3, 3), kernel_initializer=glorot_init, use_bias=True, padding='same', activation='tanh')(y)
    else:
        ys = []
        torgbs = []


        for i, block_size in enumerate(block_sizes):
            y = cond_resblock(y, cls, kernel_size=(3, 3), resample="UP", nfilters=block_size, number_of_classes=number_of_classes,
                          name='Generator.' + str(i), norm=norm, is_first=False, conv_layer=Conv2D, cond_conv_layer=ConditionalConv11,
                          cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                          cond_shortcut=conditional_shortcut, uncond_shortcut=unconditional_shortcut)
            ys.append(y)
            torgb = Activation('relu')(y)
            torgb = Conv2D(filters=output_channels, kernel_size=(1, 1), activation='tanh', name='ToRGB.' + str(i))(torgb)
            torgbs.append(torgb)

        if progressive_stage % 2 == 0:
            output = torgbs[progressive_stage / 2]
        else:
            small = torgbs[progressive_stage / 2]
            small = UpSampling2D(size=(2, 2))(small)
            large = torgbs[progressive_stage / 2 + 1]

            diff = Subtract()([small, large])
            diff = Mul(progressive_iters_per_stage, True)(diff)
            output = Add()([large, diff])

            size_mul = 2 ** (len(block_sizes) - ((progressive_stage + 1) / 2) - 1)

            output = UpSampling2D(size=(size_mul, size_mul))(output)


    no_lables = (not conditional_bn) and (not conditional_bottleneck) and (not concat_cls) and (not conditional_shortcut)
    if no_lables:
        return Model(inputs=[inp], outputs=output)
    else:
        return Model(inputs=[inp, cls], outputs=output)
