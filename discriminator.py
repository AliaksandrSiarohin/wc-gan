from keras.models import Input, Model
from keras.layers import Dense, Activation, Conv2D, GlobalAveragePooling2D, Lambda, Dropout, Flatten
from keras.layers import Add, Embedding

from gan.conditional_layers import cond_resblock, ConditionalConv11, cond_dcblock
from gan.spectral_normalized_layers import SNConv2D, SNDense, SNConditionalConv11, SNEmbeding
from gan.layer_utils import glorot_init, GlobalSumPooling2D
from functools import partial
import keras.backend as K

from generator import create_norm


def make_discriminator(input_image_shape, input_cls_shape=(1, ), block_sizes=(128, 128, 128, 128),
                       resamples=('DOWN', "DOWN", "SAME", "SAME"),
                       number_of_classes=10, type='AC_GAN', norm='n', after_norm='n', spectral=False,
                       fully_diff_spectral=False, spectral_iterations=1, conv_singular=True,
                       sum_pool=False, dropout=False, arch='res'):

    assert arch in ['res', 'dcgan']
    assert len(block_sizes) == len(resamples)
    x = Input(input_image_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if spectral:
        conv_layer = partial(SNConv2D, conv_singular=conv_singular,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        cond_conv_layer = partial(SNConditionalConv11,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        dence_layer = partial(SNDense,
                              fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
        emb_layer = partial(SNEmbeding, fully_diff_spectral=fully_diff_spectral, spectral_iterations=spectral_iterations)
    else:
        conv_layer = Conv2D
        cond_conv_layer = ConditionalConv11
        dence_layer = Dense
        emb_layer = Embedding


    norm_layer = create_norm(norm=norm, after_norm=after_norm, cls=cls, number_of_classes=number_of_classes,
                             conditional_conv_layer=cond_conv_layer, uncoditional_conv_layer=conv_layer)

    y = x
    i = 0
    for block_size, resample in zip(block_sizes, resamples):
        if arch == 'res':
            y = cond_resblock(y, kernel_size=(3, 3), resample=resample, nfilters=block_size,
                              name='Discriminator.' + str(i), norm=norm_layer, is_first=(i == 0), conv_layer=conv_layer)
            i += 1
        else:
            y = cond_dcblock(y, kernel_size=(3, 3) if resample == "SAME" else (4, 4), resample=resample, nfilters=block_size,
                              name='Discriminator.' + str(i), norm=norm_layer, is_first=(i == 0), conv_layer=conv_layer)
            i += 1

    y = Activation('relu')(y)

    if arch == 'res':
        if sum_pool:
            y = GlobalSumPooling2D()(y)
        else:
            y = GlobalAveragePooling2D()(y)
    else:
        y = Flatten()(y)

    if dropout != 0:
        y = Dropout(dropout)(y)

    if type == 'AC_GAN':
        cls_out = Dense(units=number_of_classes, use_bias=True, kernel_initializer=glorot_init)(y)
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=x, outputs=[out, cls_out])
    elif type == "PROJECTIVE":
        emb = emb_layer(input_dim = number_of_classes, output_dim = block_sizes[-1])(cls)
        phi = Lambda(lambda inp: K.sum(inp[1] * K.expand_dims(inp[0], axis=1), axis=2), output_shape=(1, ))([y, emb])
        psi = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        out = Add()([phi, psi])
        return Model(inputs=[x,cls], outputs=[out])
    elif type is None:
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x], outputs=[out])
