from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Add, Embedding, Concatenate, UpSampling2D, AveragePooling2D, Subtract

from gan.conditional_layers import cond_resblock, ConditionalConv11, ConditionalDense
from gan.spectral_normalized_layers import SNConv2D, SNDense, SNConditionalConv11, SNCondtionalDense
from gan.layer_utils import glorot_init
from generator import Mul


def make_discriminator(input_image_shape, input_cls_shape=(1, ), block_sizes=(128, 128, 128, 128),
                       resamples=('DOWN', "DOWN", "SAME", "SAME"), number_of_classes=10,
                       type='AC_GAN', norm=False, spectral=False,
                       conditional_bottleneck=False, unconditional_bottleneck=False,
                       conditional_shortcut=False, unconditional_shortcut=True,
                       progressive=False, progressive_stage=0, progressive_iters_per_stage=10000):

    assert conditional_shortcut or unconditional_shortcut
    assert len(block_sizes) == len(resamples)
    x = Input(input_image_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if spectral:
        conv_layer = SNConv2D
        cond_conv_layer = SNConditionalConv11
        dence_layer = SNDense
        cond_dence_layer = SNCondtionalDense
    else:
        conv_layer = Conv2D
        cond_conv_layer = ConditionalConv11
        dence_layer = Dense
        cond_dence_layer = ConditionalDense

    norm = BatchNormalization if norm else None

    if not progressive:
        y = x
        i = 0
        for block_size, resample in zip(block_sizes, resamples):
            y = cond_resblock(y, cls, kernel_size=(3, 3), resample=resample, nfilters=block_size,
                              number_of_classes=number_of_classes, name='Discriminator.' + str(i), norm=norm,
                              is_first=False, conv_layer=conv_layer, cond_conv_layer=cond_conv_layer,
                              cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                              cond_shortcut=conditional_shortcut, uncond_shortcut=unconditional_shortcut)
            i += 1
    else:
        size_drop = 2 ** (len(block_sizes) - ((progressive_stage + 1) / 2) - 1)
        print (size_drop)
        y = AveragePooling2D(pool_size=(size_drop, size_drop))(x)

        current_block = (progressive_stage + 1) / 2


        def from_rgb_for_block(block_num):
             if block_num == len(block_sizes) - 1:
                 return Conv2D(filters=input_image_shape[-1], kernel_size=(1, 1), name='FromRGB.' + str(block_num))
             else:
                 return Conv2D(filters=block_sizes[block_num +1], kernel_size=(1, 1), name='FromRGB.' + str(block_num))

        if progressive_stage % 2 == 0:
            y = from_rgb_for_block(current_block)(y)
        else:
            y_small = AveragePooling2D(pool_size=(2, 2))(y)
            y_small = from_rgb_for_block(current_block - 1)(y_small)
            y_large = from_rgb_for_block(current_block)(y)

            y_large = cond_resblock(y_large, cls, kernel_size=(3, 3), resample='DOWN', nfilters=block_sizes[current_block],
                              number_of_classes=number_of_classes, name='Discriminator.' + str(current_block), norm=norm,
                              is_first=False, conv_layer=conv_layer, cond_conv_layer=cond_conv_layer,
                              cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                              cond_shortcut=conditional_shortcut, uncond_shortcut=unconditional_shortcut)

            diff = Subtract()([y_small, y_large])
            diff = Mul(number_of_iters_per_stage=progressive_iters_per_stage, update_count=False)(diff)
            y = Add()([y_large, diff])
            current_block -= 1

        for _ in range(current_block + 1):
            y = cond_resblock(y, cls, kernel_size=(3, 3), resample='DOWN', nfilters=block_sizes[current_block],
                              number_of_classes=number_of_classes, name='Discriminator.' + str(current_block), norm=norm,
                              is_first=False, conv_layer=conv_layer, cond_conv_layer=cond_conv_layer,
                              cond_bottleneck=conditional_bottleneck, uncond_bottleneck=unconditional_bottleneck,
                              cond_shortcut=conditional_shortcut, uncond_shortcut=unconditional_shortcut)
            current_block -= 1


    y = Activation('relu')(y)
    y = GlobalAveragePooling2D()(y)

    if type == 'AC_GAN':
        cls_out = Dense(units=number_of_classes, use_bias=True, kernel_initializer=glorot_init)(y)
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)

        return Model(inputs=x, outputs=[out, cls_out])
    elif type == "PROJECTIVE":
        phi = cond_dence_layer(units=1, number_of_classes=number_of_classes,
                               use_bias=True, kernel_initializer=glorot_init)([y,cls])
        psi = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        out = Add()([phi, psi])
        return Model(inputs=[x,cls], outputs=[out])
    elif type is None:
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x], outputs=[out])
    else:
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x,cls], outputs=[out])
