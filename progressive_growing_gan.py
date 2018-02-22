from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, Add, Embedding, Concatenate, UpSampling2D, Layer, Subtract, AveragePooling2D

from gan.cmd import parser_with_default_args
from gan.train import Trainer
from gan.spectral_normalized_layers import SNConv2D, SNDense, SNConditionalConv11, SNCondtionalDense
from gan.ac_gan import AC_GAN
from gan.projective_gan import ProjectiveGAN
from gan.gan import GAN

from keras import backend as K

from gan.conditional_layers import ConditionalInstanceNormalization, glorot_init, ConditionalConv11, cond_resblock, ConditionalDense
from scorer import compute_scores
import numpy as np
import json
from keras.optimizers import Adam
from functools import partial

import os

iter_count = K.variable(0, dtype='int32', name='alpha')


class Mul(Layer):
    def __init__(self, number_of_iters_per_stage=10000, update_count=True, **kwargs):
        super(Mul, self).__init__(**kwargs)
        self.number_of_iters_per_stage = number_of_iters_per_stage
        self.update_count = update_count

    def call(self, inputs, **kwargs):
        if self.update_count:
            self.add_update(K.update_add(iter_count, 1))
        return (1.0 - K.cast(iter_count, dtype='float32') / self.number_of_iters_per_stage) * inputs


def make_generator(stage, input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ), block_sizes=(128, 128, 128),
                   first_block_shape=(4, 4, 128), number_of_classes=10, type="CONCAT", unconditional_bottleneck=False,
                   number_of_iters_per_stage=10000):
    assert type in [None, 'CONCAT', "COND_BN", "BOTTLENECK"]
    inp = Input(input_noise_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if type == "CONCAT":
        y = Embedding(input_dim=number_of_classes, output_dim=block_sizes[0])(cls)
        y = Concatenate(axis=-1)([y, inp])
    else:
        y = inp


    y = Dense(np.prod(first_block_shape), kernel_initializer=glorot_init, name='Dense.0')(y)
    y = Reshape(first_block_shape)(y)

    conditional_bottleneck = ConditionalConv11 if type == "BOTTLENECK" else None
    unconditional_bottleneck = Conv2D if unconditional_bottleneck else None

    if type == "COND_BN":
        norm = lambda axis, name: (lambda inp: ConditionalInstanceNormalization(number_of_classes=number_of_classes,
                                                                                axis=axis, name=name)([inp, cls]))
    else:
        norm = BatchNormalization

    ys = []
    torgbs = []


    for i, block_size in enumerate(block_sizes):
        y = cond_resblock(y, cls, kernel_size=(3, 3), resample="UP", nfilters=block_size, number_of_classes=number_of_classes,
                          norm=norm, is_first=False, conv_shortcut=True, conv_layer=Conv2D,
                          cond_bottleneck_layer=conditional_bottleneck, uncond_bottleneck_layer=unconditional_bottleneck,
                          name='GeneratorBlock.' + str(i))
        ys.append(y)
        torgb = Activation('relu')(y)
        torgb = Conv2D(filters=output_channels, kernel_size=(1, 1), activation='tanh', name='ToRGB.' + str(i))(torgb)
        torgbs.append(torgb)


    if stage % 2 == 0:
        output = torgbs[stage / 2]
    else:
        small = torgbs[stage / 2]
        small = UpSampling2D(size=(2, 2))(small)
        large = torgbs[stage / 2 + 1]

        diff = Subtract()([small, large])
        diff = Mul(number_of_iters_per_stage, True)(diff)
        output = Add()([large, diff])

    size_mul = 2 ** (len(block_sizes) - ((stage + 1) / 2) - 1)

    output = UpSampling2D(size=(size_mul, size_mul))(output)

    if type is not None:
        return Model(inputs=[inp, cls], outputs=output)
    else:
        return Model(inputs=[inp], outputs=output)


def make_discriminator(stage, input_image_shape, input_cls_shape=(1, ), block_sizes=(256, 128, 64),
                       number_of_classes=10, type='AC_GAN', norm=False, spectral=False, unconditional_bottleneck=False,
                       number_of_iters_per_stage=10000):
    assert type in [None, 'AC_GAN', 'PROJECTIVE', 'BOTTLENECK']
    x = Input(input_image_shape)
    cls = Input(input_cls_shape, dtype='int32')

    conditional_bottleneck = True if type == "BOTTLENECK" else False

    if spectral:
        conv_layer = SNConv2D
        conditional_bottleneck = SNConditionalConv11 if conditional_bottleneck else None
        unconditional_bottleneck = SNConv2D if unconditional_bottleneck else None
        dence_layer = SNDense
        cond_dence_layer = SNCondtionalDense
    else:
        conv_layer = Conv2D
        conditional_bottleneck = ConditionalConv11 if conditional_bottleneck else None
        unconditional_bottleneck = Conv2D if unconditional_bottleneck else None
        dence_layer = Dense
        cond_dence_layer = ConditionalDense
    norm = BatchNormalization if norm else None

    size_drop = 2 ** (len(block_sizes) - ((stage + 1) / 2) - 1)
    print (size_drop)
    y = AveragePooling2D(pool_size=(size_drop, size_drop))(x)

    current_block = (stage + 1) / 2


    def from_rgb_for_block(block_num):
         if block_num == len(block_sizes) - 1:
             return Conv2D(filters=input_image_shape[-1], kernel_size=(1, 1), name='FromRGB.' + str(block_num))
         else:
             return Conv2D(filters=block_sizes[block_num +1], kernel_size=(1, 1), name='FromRGB.' + str(block_num))

    if stage % 2 == 0:
        y = from_rgb_for_block(current_block)(y)
    else:
        y_small = AveragePooling2D(pool_size=(2, 2))(y)
        y_small = from_rgb_for_block(current_block - 1)(y_small)
        y_large = from_rgb_for_block(current_block)(y)

        y_large = cond_resblock(y_large, cls, kernel_size=(3, 3), resample='DOWN', nfilters=block_sizes[current_block],
                                number_of_classes=number_of_classes, norm=norm, is_first=True,
                                conv_shortcut=True, conv_layer=conv_layer, name = 'Dicriminator.' + str(current_block),
                                cond_bottleneck_layer=conditional_bottleneck, uncond_bottleneck_layer=unconditional_bottleneck)

        diff = Subtract()([y_small, y_large])
        diff = Mul(number_of_iters_per_stage=number_of_iters_per_stage, update_count=False)(diff)
        y = Add()([y_large, diff])
        current_block -= 1

    for _ in range(current_block + 1):
        y = cond_resblock(y, cls, kernel_size=(3, 3), resample='DOWN', nfilters=block_sizes[current_block],
                          number_of_classes=number_of_classes, norm=norm, is_first=True,
                          conv_shortcut=True, conv_layer=conv_layer, name = 'Dicriminator.' + str(current_block),
                          cond_bottleneck_layer=conditional_bottleneck, uncond_bottleneck_layer=unconditional_bottleneck)
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
    elif type == 'BOTTLENECK':
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x,cls], outputs=[out])
    else:
        out = dence_layer(units=1, use_bias=True, kernel_initializer=glorot_init)(y)
        return Model(inputs=[x], outputs=[out])


def compile_and_run_progresive(dataset, args):
    additional_info = json.dumps(vars(args))

    args.generator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
    args.discriminator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)

    log_file = os.path.join(args.output_dir, 'log.txt')

    number_of_stages = 5 if args.dataset == 'mnist' else 7
    image_shape = (28, 28, 1) if args.dataset == 'mnist' else (32, 32, 3)

    def run_stage(stage, generator_checkpoint, discriminator_checkpoint):
        generator = make_generator(stage, output_channels=1 if args.dataset == 'mnist' else 3,
                                   block_sizes=(128, 128) if args.dataset == 'mnist' else (128, 128, 128),
                                   first_block_shape=(7, 7, 128) if args.dataset == 'mnist' else (4, 4, 128),
                                   type=args.generator_type,
                                   unconditional_bottleneck=bool(args.uncoditional_bottleneck),
                                   number_of_iters_per_stage=args.epochs_per_progresive_stage * 1000)

        discriminator = make_discriminator(stage, input_image_shape=image_shape,
                                           block_sizes=(256, 128) if args.dataset == 'mnist' else (256, 128, 64),
                                           type=args.discriminator_type,
                                           norm=args.bn_in_discriminator,
                                           spectral=args.spectral,
                                           unconditional_bottleneck=args.uncoditional_bottleneck,
                                           number_of_iters_per_stage=args.epochs_per_progresive_stage * 1000)

        generator.summary()
        discriminator.summary()
        print ("Print current iter count %s:" % K.get_value(iter_count))

        if generator_checkpoint is not None:
            generator.load_weights(generator_checkpoint, by_name=True)

        if discriminator_checkpoint is not None:
            discriminator.load_weights(discriminator_checkpoint, by_name=True)

        if args.dataset == 'mnist':
            at_store_checkpoint_hook = partial(compute_scores, image_shape=image_shape, log_file=log_file,
                                            generator=generator, dataset=dataset, compute_inception=False, compute_fid=False,
                                            additional_info=additional_info)
        else:
            at_store_checkpoint_hook = partial(compute_scores, image_shape=image_shape, log_file=log_file,
                                           generator=generator, dataset=dataset, compute_inception=False, compute_fid=False,
                                           additional_info=additional_info)

        if args.phase=='train':
            GANS = {None:GAN, 'AC_GAN':AC_GAN, 'PROJECTIVE':ProjectiveGAN, 'BOTTLENECK':ProjectiveGAN}
            gan = GANS[args.discriminator_type](generator=generator, discriminator=discriminator, **vars(args))

            dataset_type = args.dataset
            del args.dataset
            args.number_of_epochs = args.epochs_per_progresive_stage
            trainer = Trainer(dataset, gan, at_store_checkpoint_hook=at_store_checkpoint_hook,  **vars(args))
            trainer.train()
            args.dataset = dataset_type
        else:
            at_store_checkpoint_hook()

        print ("Current iter count %s:" % K.get_value(iter_count))
        K.set_value(iter_count, 0)

        if not os.path.exists(args.tmp_progresive_checkpoints_dir):
            os.makedirs(args.tmp_progresive_checkpoints_dir)
        generator_checkpoint = os.path.join(args.tmp_progresive_checkpoints_dir, 'generator.' + str(stage))
        discriminator_checkpoint = os.path.join(args.tmp_progresive_checkpoints_dir, 'discirminator.' + str(stage))

        generator.save_weights(generator_checkpoint)
        discriminator.save_weights(discriminator_checkpoint)

        return generator_checkpoint, discriminator_checkpoint

    generator_checkpoint = None
    discriminator_checkpoint = None
    if args.phase == 'train':
        for stage in range(number_of_stages):
            generator_checkpoint, discriminator_checkpoint = run_stage(stage, generator_checkpoint, discriminator_checkpoint)
    else:
        run_stage(number_of_stages - 1, args.generator_checkpoint, args.discriminator_checkpoint)


