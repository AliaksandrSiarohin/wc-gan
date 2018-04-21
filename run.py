from keras.optimizers import Adam

from gan.dataset import LabeledArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
from gan.ac_gan import AC_GAN
from gan.projective_gan import ProjectiveGAN
from gan.gan import GAN
from gan.conditional_layers import ConditionalAdamOptimizer

import os
import json
from functools import partial
from scorer import compute_scores
from time import time
from argparse import Namespace

from generator import make_generator
from discriminator import make_discriminator
from keras.utils import plot_model

from keras import backend as K
from keras.backend import tf as ktf


def get_dataset(dataset, batch_size, supervised = False, noise_size=(128, )):
    assert dataset in ['mnist', 'cifar10', 'cifar100', 'fashion-mnist', 'stl10']

    if dataset == 'mnist':
        from keras.datasets import mnist
        (X, y), (X_test, y_test) = mnist.load_data()
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    elif dataset == 'cifar10':
        from cifar10 import load_data
        (X, y), (X_test, y_test) = load_data()
    elif dataset == 'cifar100':
        from cifar100 import load_data
        (X, y), (X_test, y_test) = load_data()
    elif dataset == 'fashion-mnist':
        from fashion_mnist import load_data
        (X, y), (X_test, y_test) = load_data()
        X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2], 1))
    elif dataset == 'stl10':
        from stl10 import load_data
        (X, y), (X_test, y_test) = load_data()
        assert not supervised

    return LabeledArrayDataset(X=X, y=y if supervised else None, X_test=X_test, y_test=y_test,
                               batch_size=batch_size, noise_size=noise_size)


def compile_and_run(dataset, args, generator_params, discriminator_params):
    additional_info = json.dumps(vars(args))

    args.generator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
    args.discriminator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)

    log_file = os.path.join(args.output_dir, 'log.txt')

    at_store_checkpoint_hook = partial(compute_scores, image_shape=args.image_shape, log_file=log_file,
                                       dataset=dataset, images_inception=args.samples_inception,
                                       images_fid=args.samples_fid, additional_info=additional_info,
                                       cache_file=args.fid_cache_file)

    lr_decay_schedule_generator, lr_decay_schedule_discriminator = get_lr_decay_schedule(args)

    generator_checkpoint = args.generator_checkpoint
    discriminator_checkpoint = args.discriminator_checkpoint

    generator = make_generator(**vars(generator_params))
    discriminator = make_discriminator(**vars(discriminator_params))

    generator.summary()
    discriminator.summary()

    if generator_checkpoint is not None:
        generator.load_weights(generator_checkpoint)#, by_name=True)

    if discriminator_checkpoint is not None:
        discriminator.load_weights(discriminator_checkpoint, by_name=True)

    hook = partial(at_store_checkpoint_hook, generator=generator)

    if args.phase == 'train':
        GANS = {None:GAN, 'AC_GAN':AC_GAN, 'PROJECTIVE':ProjectiveGAN}
        gan = GANS[args.gan_type](generator=generator, discriminator=discriminator,
                                                lr_decay_schedule_discriminator = lr_decay_schedule_discriminator,
                                                lr_decay_schedule_generator = lr_decay_schedule_generator,
                                                **vars(args))
        trainer = Trainer(dataset, gan, at_store_checkpoint_hook=hook,**vars(args))
        trainer.train()
    else:
        hook(0)


def get_lr_decay_schedule(args):
    number_of_iters_generator = 1000. * args.number_of_epochs
    number_of_iters_discriminator = 1000. * args.number_of_epochs * args.training_ratio

    if args.lr_decay_schedule is None:
        lr_decay_schedule_generator = lambda iter: 1.
        lr_decay_schedule_discriminator = lambda iter: 1.
    elif args.lr_decay_schedule == 'linear':
        lr_decay_schedule_generator = lambda iter: K.maximum(0., 1. - K.cast(iter, 'float32') / number_of_iters_generator)
        lr_decay_schedule_discriminator = lambda iter: K.maximum(0., 1. - K.cast(iter, 'float32') / number_of_iters_discriminator)
    elif args.lr_decay_schedule == 'half-linear':
        lr_decay_schedule_generator = lambda iter: ktf.where(
                                K.less(iter, K.cast(number_of_iters_generator / 2, 'int64')),
                                ktf.maximum(0., 1. - (K.cast(iter, 'float32') / number_of_iters_generator)), 0.5)
        lr_decay_schedule_discriminator = lambda iter: ktf.where(
                                K.less(iter, K.cast(number_of_iters_discriminator / 2, 'int64')),
                                ktf.maximum(0., 1. - (K.cast(iter, 'float32') / number_of_iters_discriminator)), 0.5)
    elif args.lr_decay_schedule == 'linear-end':
        decay_at = 0.9

        number_of_iters_until_decay_generator = number_of_iters_generator * decay_at
        number_of_iters_until_decay_discriminator = number_of_iters_discriminator * decay_at

        number_of_iters_after_decay_generator = number_of_iters_generator * (1 - decay_at)
        number_of_iters_after_decay_discriminator = number_of_iters_discriminator * (1 - decay_at)


        lr_decay_schedule_generator = lambda iter: ktf.where(
                                K.greater(iter, K.cast(number_of_iters_until_decay_generator, 'int64')),
                                ktf.maximum(0., 1. - (K.cast(iter, 'float32') - number_of_iters_until_decay_generator) / number_of_iters_after_decay_generator), 1)
        lr_decay_schedule_discriminator = lambda iter: ktf.where(
                                K.greater(iter, K.cast(number_of_iters_until_decay_discriminator, 'int64')),
                                ktf.maximum(0., 1. - (K.cast(iter, 'float32') - number_of_iters_until_decay_discriminator) / number_of_iters_after_decay_discriminator), 1)
    else:
        assert False

    return lr_decay_schedule_generator, lr_decay_schedule_discriminator


def get_generator_params(args):
    params = Namespace()
    params.output_channels = 1 if args.dataset.endswith('mnist') else 3
    params.input_cls_shape = (1, )

    first_block_w = (7 if args.dataset.endswith('mnist') else (6 if args.dataset == 'stl10' else 4))
    params.first_block_shape = (first_block_w, first_block_w, args.generator_filters)
    if args.arch == 'res':
        params.block_sizes = tuple([args.generator_filters] * 2) if args.dataset.endswith('mnist') else tuple([args.generator_filters] * 3)
        params.resamples = ("UP", "UP") if args.dataset.endswith('mnist') else ("UP", "UP", "UP")
    else:
        params.block_sizes = ([args.generator_filters, args.generator_filters / 2] if args.dataset.endswith('mnist')
                              else [args.generator_filters, args.generator_filters / 2, args.generator_filters / 4])
        params.resamples = ("UP", "UP") if args.dataset.endswith('mnist') else ("UP", "UP", "UP")
    params.number_of_classes = 10 if args.dataset != 'cifar100' else 100

    params.concat_cls = args.generator_concat_cls

    params.renorm_for_decor = args.generator_renorm_for_decor

    params.block_norm = args.generator_block_norm
    params.block_after_norm = args.generator_block_after_norm

    params.last_norm = args.generator_last_norm
    params.last_after_norm = args.generator_last_after_norm

    params.gan_type = args.gan_type
    
    params.arch = args.arch
    return params


def get_discriminator_params(args):
    params = Namespace()
    params.input_image_shape = args.image_shape
    params.input_cls_shape = (1, )
    if args.arch == 'res':
        params.block_sizes = tuple([args.discriminator_filters] * 4)
        params.resamples = ('DOWN', "DOWN", "SAME", "SAME")
    else:
        params.block_sizes = [args.discriminator_filters / 8, args.discriminator_filters / 8,
                              args.discriminator_filters / 4, args.discriminator_filters / 4,
                              args.discriminator_filters / 2, args.discriminator_filters / 2,
                              args.discriminator_filters]
        params.resamples = ('SAME', "DOWN", "SAME", "DOWN", "SAME", "DOWN", "SAME")
    params.number_of_classes = 10 if args.dataset != 'cifar100' else 100

    params.norm = args.discriminator_norm
    params.after_norm = args.discriminator_after_norm

    params.spectral = args.spectral
    params.fully_diff_spectral = args.fully_diff_spectral
    params.spectral_iterations = args.spectral_iterations
    params.conv_singular = args.conv_singular

    params.type = args.gan_type

    params.sum_pool = args.sum_pool
    params.dropout = args.discriminator_dropout

    params.arch = args.arch

    return params


def main():
    parser = parser_with_default_args()
    parser.add_argument("--phase", choices=['train', 'test'], default='train')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate")
    parser.add_argument("--beta1", default=0, type=float, help='Adam parameter')
    parser.add_argument("--beta2", default=0.9, type=float, help='Adam parameter')
    parser.add_argument("--dataset", default='cifar10', choices=['mnist', 'cifar10', 'cifar100', 'fashion-mnist', 'stl10'],
                        help='Dataset to train on')
    parser.add_argument("--arch", default='res', choices=['res', 'dcgan'], help="Gan architecture resnet or dcgan.")

    parser.add_argument("--spectral", default=0, type=int, help='Use spectral norm in discriminator')
    parser.add_argument("--fully_diff_spectral", default=0, type=int, help='Fully difirentiable spectral normalization')
    parser.add_argument("--spectral_iterations", default=1, type=int, help='Number of iteration per spectral update')
    parser.add_argument("--conv_singular", default=0, type=int, help='Singular convolution layer')

    parser.add_argument("--generator_block_norm", default='u', choices=['n', 'b', 'd'],
                        help='Normalization in generator resblock. b - batch, d - decorelation, n - none.')
    parser.add_argument("--generator_block_after_norm", default='n', choices=['ccs', 'ucs', 'uccs', 'cconv', 'uconv', 'ucconv','ccsuconv', 'n'],
                        help="Layer after normalization")

    parser.add_argument("--generator_last_norm", default='b', choices=['n', 'b', 'd'],
                        help='Batch normalization in generator last. cb - conditional batch,'
                             ' ub - unconditional batch, n - none.'
                             'conv - conv11 after uncoditional, d - decorelation.')
    parser.add_argument("--generator_last_after_norm", default='n', choices=['ccs', 'ucs', 'uccs', 'cconv', 'uconv', 'ucconv', 'ccsuconv', 'n'],
                        help="Layer after normalization")

    parser.add_argument("--generator_renorm_for_decor", default=0, type=int, help='Renorm for decorelation normalization')
    parser.add_argument("--generator_concat_cls", default=0, type=int, help='Concat labels to noise in genrator')

    parser.add_argument("--generator_filters", default=128, type=int, help='Number of filters in generator block')

    parser.add_argument("--gan_type", default=None, choices=[None, 'AC_GAN', 'PROJECTIVE'],
                        help='Type of gan to use. None for unsuperwised.')

    parser.add_argument("--discriminator_norm", default='n', choices=['n', 'b', 'd'],
                        help='Normalization in generator resblock. b - batch, d - decorelation, n - none.')

    parser.add_argument("--discriminator_after_norm", default='n',
                        choices=['ccs', 'ucs', 'uccs', 'cconv', 'uconv', 'ucconv', 'ccsuconv', 'n'],
                        help="Layer after normalization")

    parser.add_argument("--discriminator_filters", default=128, type=int, help='Number of filters in discriminator_block')
    parser.add_argument("--discriminator_dropout", type=float, default=0, help="Use dropout in discriminator")

    parser.add_argument("--samples_inception", default=50000, type=int, help='Samples for inception, 0 - no compute inception')
    parser.add_argument("--samples_fid", default=10000, type=int, help="Samples for FID, 0 - no compute FID")

    parser.add_argument("--lr_decay_schedule", default=None, choices=[None, 'linear', 'half-linear', 'linear-end'],
                        help='Learnign rate decay schedule. None - no decay. '
                             'linear - linear decay to zero. half-linear - linear decay to 0.5'
                             'linear-end constant until 0.9, then linear decay to 0')

    parser.add_argument("--sum_pool", default=1, type=int, help='Use sum or average pooling')

    args = parser.parse_args()

    dataset = get_dataset(dataset=args.dataset,
                          batch_size=args.batch_size,
                          supervised=args.gan_type is not None)

    args.output_dir = "output/%s_%s" % (args.dataset, time())
    args.checkpoints_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)

    args.image_shape = (28, 28, 1) if args.dataset.endswith('mnist') else (32, 32, 3)
    args.fid_cache_file = "output/%s_fid.npz" % args.dataset

    discriminator_params = get_discriminator_params(args)
    generator_params = get_generator_params(args)

    del args.dataset

    compile_and_run(dataset, args, generator_params, discriminator_params)


if __name__ == "__main__":
    main()
