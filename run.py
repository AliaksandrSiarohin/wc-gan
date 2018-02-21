from keras.models import Input, Model
from keras.layers import Dense, Reshape, Activation, Conv2D, GlobalAveragePooling2D, BatchNormalization, Add, Embedding, Concatenate
from keras.optimizers import Adam

from gan.dataset import LabeledArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
from gan.inception_score import get_inception_score
from gan.fid import calculate_fid_given_arrays
from gan.ac_gan import AC_GAN
from gan.projective_gan import ProjectiveGAN
from gan.spectral_normalized_layers import SNConv2D, SNDense, SNConditionalConv11, SNCondtionalDense
from gan.gan import GAN

import numpy as np
from gan.layer_utils import glorot_init

from gan.conditional_layers import ConditionalInstanceNormalization, cond_resblock, ConditionalConv11, ConditionalDense
from tqdm import tqdm
import os
import json
from functools import partial


def make_generator(input_noise_shape=(128,), output_channels=3, input_cls_shape=(1, ), block_sizes=(128, 128, 128),
                   first_block_shape=(4, 4, 128), number_of_classes=10, type="CONCAT", unconditional_bottleneck=False):
    assert type in [None, 'CONCAT', "COND_BN", "BOTTLENECK"]
    inp = Input(input_noise_shape)
    cls = Input(input_cls_shape, dtype='int32')

    if type == "CONCAT":
        y = Embedding(input_dim=number_of_classes, output_dim=block_sizes[0])(cls)
        y = Concatenate(axis=-1)([y, inp])
    else:
        y = inp

    y = Dense(np.prod(first_block_shape), kernel_initializer=glorot_init)(y)
    y = Reshape(first_block_shape)(y)

    conditional_bottleneck = ConditionalConv11 if type == "BOTTLENECK" else None
    unconditional_bottleneck = Conv2D if unconditional_bottleneck else None

    if type == "COND_BN":
        norm = lambda axis: (lambda inp: ConditionalInstanceNormalization(number_of_classes=10, axis=axis)([inp, cls]))
    else:
        norm = BatchNormalization

    for block_size in block_sizes:
        y = cond_resblock(y, cls, kernel_size=(3, 3), resample="UP", nfilters=block_size, number_of_classes=number_of_classes,
                      norm=norm, is_first=False, conv_shortcut=True, conv_layer=Conv2D,
                      cond_bottleneck_layer=conditional_bottleneck, uncond_bottleneck_layer=unconditional_bottleneck)

    y = BatchNormalization(axis=-1)(y)
    y = Activation('relu')(y)
    y = Conv2D(output_channels, (3, 3), kernel_initializer=glorot_init, use_bias=True, padding='same', activation='tanh')(y)

    if type is not None:
        return Model(inputs=[inp, cls], outputs=y)
    else:
        return Model(inputs=[inp], outputs=y)



def make_discriminator(input_image_shape, input_cls_shape=(1, ), block_sizes=(128, 128, 128, 128),
                       resamples = ('DOWN', "DOWN", "SAME", "SAME"), number_of_classes=10,
                       type='AC_GAN', norm=False, spectral=False,
                       unconditional_bottleneck=False):
    assert type in [None, 'AC_GAN', 'PROJECTIVE', 'BOTTLENECK']
    assert len(block_sizes) == len(resamples)
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

    y = x
    is_first = True
    for block_size, resample in zip(block_sizes, resamples):
        y = cond_resblock(y, cls, kernel_size=(3, 3), resample=resample, nfilters=block_size,
                          number_of_classes=number_of_classes, norm=norm, is_first=is_first,
                          conv_shortcut=True, conv_layer=conv_layer,
                          cond_bottleneck_layer=conditional_bottleneck, uncond_bottleneck_layer=unconditional_bottleneck)
        is_first = False

    y = Activation('relu')(y)
    y = GlobalAveragePooling2D()(y)

    if type == 'AC_GAN':
        cls_out = dence_layer(units=number_of_classes, use_bias=True, kernel_initializer=glorot_init)(y)
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


def get_dataset(dataset, batch_size, supervised = False, noise_size = (128, )):
    assert dataset in ['mnist', 'cifar10']

    if dataset == 'mnist':
        from keras.datasets import mnist
        (X, y), (X_test, y_test) = mnist.load_data()
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    elif dataset == 'cifar10':
        from cifar10 import load_data
        (X, y), (X_test, y_test) = load_data()

    return LabeledArrayDataset(X=X, y=y if supervised else None, batch_size=batch_size, noise_size=noise_size)


def compute_scores(epoch, image_shape, generator, dataset, number_of_images=50000, compute_inception=True, compute_fid=True,
                   log_file=None):
    if not (compute_inception or compute_fid):
        return
    images = np.empty((number_of_images, ) + image_shape)
    previous_batsh_size = dataset._batch_size
    dataset._batch_size = 100
    for i in tqdm(range(0, 50000, 100)):
        g_s = dataset.next_generator_sample_test()
        images[i:(i+100)] = generator.predict(g_s)
    images *= 127.5
    images += 127.5
    dataset._batch_size = previous_batsh_size

    if compute_inception:
        str = "INCEPTION SCORE: %s" % get_inception_score(images)
        print (str)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print >>f, ("Epoch %s" % (epoch, )) + str
    if compute_fid:
        true_images = 127.5 * dataset._X + 127.5
        str = "FID SCORE: %s" % calculate_fid_given_arrays([true_images, images])
        print (str)
        if log_file is not None:
            with open(log_file, 'a') as f:
                print >>f, ("Epoch %s" % (epoch, )) + str


def main():
    parser = parser_with_default_args()
    parser.add_argument("--phase", choices=['train', 'test'], default='train')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate")
    parser.add_argument("--beta1", default=0.5, type=float, help='Adam parameter')
    parser.add_argument("--beta2", default=0.999, type=float, help='Adam parameter')
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'cifar10'], help='Dataset to train on')
    parser.add_argument("--spectral", default=0, type=int, help='Use spectral norm in discriminator')
    parser.add_argument("--generator_type", default=None, choices=[None, "CONCAT", "COND_BN", "BOTTLENECK"],
                        help='Type of generator to use. None for unsuperwised')
    parser.add_argument("--discriminator_type", default=None, choices=[None, 'AC_GAN', 'PROJECTIVE', 'BOTTLENECK'],
                        help='Type of generator to use. None for unsuperwised')
    parser.add_argument("--uncoditional_bottleneck", default=0, type=int)
    parser.add_argument("--bn_in_discriminator", default=0, type=int)
    args = parser.parse_args()

    dataset = get_dataset(dataset = args.dataset,
                          batch_size=args.batch_size,
                          supervised=args.generator_type is not None)

    args.output_dir = "output/%s_%s_%s" % (args.dataset, str(args.generator_type), str(args.discriminator_type))
    args.checkpoints_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)

    image_shape = (28, 28, 1) if args.dataset == 'mnist' else (32, 32, 3)

    generator = make_generator(output_channels=1 if args.dataset == 'mnist' else 3,
                               block_sizes=(128, 128) if args.dataset == 'mnist' else (128, 128, 128),
                               first_block_shape=(7, 7, 128) if args.dataset == 'mnist' else (4, 4, 128),
                               type = args.generator_type,
                               unconditional_bottleneck=bool(args.uncoditional_bottleneck))

    discriminator = make_discriminator(input_image_shape=image_shape,
                                       type=args.discriminator_type,
                                       norm=args.bn_in_discriminator,
                                       spectral=args.spectral,
                                       unconditional_bottleneck=args.uncoditional_bottleneck)

    print (generator.summary())
    print (discriminator.summary())

    if args.generator_checkpoint is not None:
        generator.load_weights(args.generator_checkpoint)
    if args.discriminator_checkpoint is not None:
        discriminator.load_weights(args.discriminator_checkpoint)

    args.generator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
    args.discriminator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)

    log_file = os.path.join(args.output_dir, 'log.txt')

    if args.dataset == 'mnist':
        at_store_checkpoint_hook = partial(compute_scores, image_shape=image_shape, log_file=log_file,
                                               generator=generator, dataset=dataset, compute_inception=False)
    else:
        at_store_checkpoint_hook = partial(compute_scores, image_shape=image_shape, log_file=log_file,
                                               generator=generator, dataset=dataset, compute_inception=False)

    if args.phase=='train':
        GANS = {None:GAN, 'AC_GAN':AC_GAN, 'PROJECTIVE':ProjectiveGAN, 'BOTTLENECK':ProjectiveGAN}
        gan = GANS[args.discriminator_type](generator=generator, discriminator=discriminator, **vars(args))

        del args.dataset
        trainer = Trainer(dataset, gan, at_store_checkpoint_hook=at_store_checkpoint_hook,  **vars(args))
        trainer.train()
    else:
        at_store_checkpoint_hook()

if __name__ == "__main__":
    main()
