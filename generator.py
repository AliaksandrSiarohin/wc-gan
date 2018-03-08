from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, Layer
from keras.layers import BatchNormalization, Add, Embedding, Concatenate, UpSampling2D, AveragePooling2D, Subtract

import numpy as np
from gan.layer_utils import glorot_init

from gan.conditional_layers import ConditinalBatchNormalization, cond_resblock, ConditionalConv11, ConditionalDepthwiseConv2D, get_separable_conv

import keras.backend as K
from functools import partial


def make_generator(input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ),
                   block_sizes=(128, 128, 128), resamples=("UP", "UP", "UP"),
                   first_block_shape=(4, 4, 128), number_of_classes=10, concat_cls=False,
                   conditional_bottleneck=False, unconditional_bottleneck=False,
                   conditional_shortcut=False, unconditional_shortcut=True,
                   conditional_bn=False, norm=True, cls_branch=False):

    assert conditional_shortcut or unconditional_shortcut

    inp = Input(input_noise_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if concat_cls:
        y = Embedding(input_dim=number_of_classes, output_dim=first_block_shape[-1])(cls)
        y = Reshape((first_block_shape[-1], ))(y)
        y = Concatenate(axis=-1)([y, inp])
    else:
        y = inp

    y = Dense(np.prod(first_block_shape), kernel_initializer=glorot_init)(y)
    y = Reshape(first_block_shape)(y)

    if norm:
        if conditional_bn:
            norm_layer = lambda axis, name: (lambda inp: ConditinalBatchNormalization(number_of_classes=number_of_classes,
                                                                                    axis=axis, name=name)([inp, cls]))
        else:
            norm_layer = BatchNormalization
    else:
        norm_layer = lambda axis, name: (lambda inp: inp)

    conv_layer_cls = partial(get_separable_conv, number_of_classes=number_of_classes, cls=cls,
                             conv_layer=ConditionalDepthwiseConv2D, conv11_layer=ConditionalConv11,
                             conditional_conv11=True, conditional_conv=True)

    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        y = cond_resblock(y, cls, kernel_size=(3, 3), resample=resample, nfilters=block_size, number_of_classes=number_of_classes,
                          name='Generator.' + str(i), norm=norm_layer, is_first=False, conv_layer=Conv2D, cond_conv_layer=ConditionalConv11,
                          cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                          cond_shortcut=conditional_shortcut, uncond_shortcut=unconditional_shortcut,
                          cls_conv=conv_layer_cls if cls_branch else None)
        i += 1

    if norm:
        y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    output = Conv2D(filters=output_channels, kernel_size=(3, 3), name='Generator.Final',
                            kernel_initializer=glorot_init, use_bias=True, padding='same')(y)
    output = Activation('tanh')(output)

    no_lables = (not conditional_bn) and (not conditional_bottleneck) and (not concat_cls) and (not conditional_shortcut) and (not cls_branch)
    if no_lables:
        return Model(inputs=[inp], outputs=output)
    else:
        return Model(inputs=[inp, cls], outputs=output)
