from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, Layer, Deconv2D
from keras.layers import BatchNormalization, Add, Embedding, Concatenate, UpSampling2D, AveragePooling2D, Subtract

import numpy as np
from gan.layer_utils import glorot_init

from gan.conditional_layers import ConditinalBatchNormalization, cond_resblock, ConditionalConv11, ConditionalDepthwiseConv2D, get_separable_conv,\
                                   DecorelationNormalization, ConditionalCenterScale, CenterScale, cond_dcblock

import keras.backend as K
from functools import partial


def create_norm(norm, after_norm, cls=None, number_of_classes=None,
                triangular_conv=False, uncoditional_conv_layer=Conv2D, conditional_conv_layer=ConditionalConv11):
    assert norm in ['n', 'b', 'd']
    assert after_norm in ['ucs', 'ccs', 'uccs', 'uconv', 'cconv', 'ucconv', 'n']

    if norm == 'n':
        norm_layer = lambda axis, name: (lambda inp: inp)
    elif norm == 'b':
        norm_layer = lambda axis, name: BatchNormalization(axis=axis, center=False, scale=False, name=name)
    elif norm == 'd':
        norm_layer = lambda axis, name: DecorelationNormalization(name=name)

    if after_norm == 'ccs':
        after_norm_layer = lambda axis, name: lambda x: ConditionalCenterScale(number_of_classes=number_of_classes,
                                                                     axis=axis, name=name)([x, cls])
    elif after_norm == 'ucs':
        after_norm_layer = lambda axis, name: lambda x: CenterScale(axis=axis, name=name)(x)
    elif after_norm == 'uccs':
        def after_norm_layer(axis, name):
            def f(x):
                c = ConditionalCenterScale(number_of_classes=number_of_classes, axis=axis, name=name + '_c')([x, cls])
                u = CenterScale(axis=axis, name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif after_norm == 'cconv':
        after_norm_layer = lambda axis, name: lambda x: conditional_conv_layer(filters=K.int_shape(x)[axis],
                                                                          number_of_classes=number_of_classes,
                                                                          name=name, triangular=triangular_conv)([x, cls])
    elif after_norm == 'uconv':
        after_norm_layer = lambda axis, name: lambda x: uncoditional_conv_layer(kernel_size=(1, 1),
                                                               filters=K.int_shape(x)[axis], name=name)(x)
    elif after_norm == 'ucconv':
        def after_norm_layer(axis, name):
            def f(x):
                c = conditional_conv_layer(number_of_classes=number_of_classes, name=name + '_c',
                                      filters=K.int_shape(x)[axis], triangular=triangular_conv)([x, cls])
                u = uncoditional_conv_layer(kernel_size=(1, 1), filters=K.int_shape(x)[axis], name=name + '_u')(x)
                out = Add(name=name + '_a')([c, u])
                return out
            return f
    elif after_norm == 'n':
        after_norm_layer = lambda axis, name: lambda x: x

    def result_norm(axis, name):
        def stack(inp):
            out = inp
            out = norm_layer(axis=axis, name=name + '_npart')(out)
            out = after_norm_layer(axis=axis, name=name + '_repart')(out)
            return out
        return stack

    return result_norm


def make_generator(input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ),
                   block_sizes=(128, 128, 128), resamples=("UP", "UP", "UP"),
                   first_block_shape=(4, 4, 128), number_of_classes=10, concat_cls=False,
                   block_norm='u', block_after_norm='cs',
                   last_norm='u', last_after_norm='cs',
                   renorm_for_decor=False, triangular_conv=False, gan_type=None,
                   arch='res'):

    assert arch in ['res', 'dcgan']
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

    block_norm_layer = create_norm(block_norm, block_after_norm, cls=cls,
                             number_of_classes=number_of_classes, triangular_conv=triangular_conv)

    last_norm_layer = create_norm(last_norm, last_after_norm, cls=cls,
                             number_of_classes=number_of_classes, triangular_conv=triangular_conv)


    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        if arch == 'res':
            y = cond_resblock(y, kernel_size=(3, 3), resample=resample,
                              nfilters=block_size, name='Generator.' + str(i),
                              norm=block_norm_layer, is_first=False)
        else:
            y = cond_dcblock(y, kernel_size=(4, 4), resample=resample,
                              nfilters=block_size, name='Generator.' + str(i),
                              norm=block_norm_layer, is_first=False, conv_layer=Deconv2D)
        i += 1

    y = last_norm_layer(axis=-1, name='Generator.BN.Final')(y)
    y = Activation('relu')(y)
    output = Conv2D(filters=output_channels, kernel_size=(3, 3), name='Generator.Final',
                            kernel_initializer=glorot_init, use_bias=True, padding='same')(y)
    output = Activation('tanh')(output)

    if gan_type is None:
        return Model(inputs=[inp], outputs=output)
    else:
        return Model(inputs=[inp, cls], outputs=output)

