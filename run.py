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


def compile_and_run(dataset, args, generator_params, discriminator_params):
    additional_info = json.dumps(vars(args))
    if not args.conditional_optimizer:
        args.generator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
        args.discriminator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
    else:
        args.generator_optimizer = ConditionalAdamOptimizer(number_of_classes=generator_params.number_of_classes,
                                                            lr=args.lr, beta_1=args.beta1, beta_2=args.beta2)
        args.discriminator_optimizer = ConditionalAdamOptimizer(number_of_classes=discriminator_params.number_of_classes,
                                                             lr=args.lr, beta_1=args.beta1, beta_2=args.beta2)

    log_file = os.path.join(args.output_dir, 'log.txt')

    at_store_checkpoint_hook = partial(compute_scores, image_shape=args.image_shape, log_file=log_file,
                                       dataset=dataset, compute_inception=args.compute_inception,
                                       compute_fid=args.compute_fid, additional_info=additional_info,
                                       number_of_images=args.samples_for_evaluation, cache_file=args.fid_cache_file)

    lr_decay_schedule_generator, lr_decay_schedule_discriminator = get_lr_decay_schedule(args)

    def run_stage(stage, generator_checkpoint, discriminator_checkpoint):
        generator = make_generator(**vars(generator_params))
        discriminator = make_discriminator(**vars(discriminator_params))

        if args.plot_model:
            plot_model(generator, os.path.join(args.output_dir, "Generator." + str(stage)))
            plot_model(generator, os.path.join(args.output_dir, "Discriminator." + str(stage)))

        if args.print_summary:
            generator.summary()
            discriminator.summary()

        if args.progressive:
            print ("Stage number %s" % (stage, ))

        if generator_checkpoint is not None:
            generator.load_weights(generator_checkpoint, by_name=True)

        if discriminator_checkpoint is not None:
            discriminator.load_weights(discriminator_checkpoint, by_name=True)

        hook = partial(at_store_checkpoint_hook, generator=generator)

        if args.phase == 'train':
            GANS = {None:GAN, 'AC_GAN':AC_GAN, 'PROJECTIVE':ProjectiveGAN, 'CLS':ProjectiveGAN}
            gan = GANS[args.gan_type](generator=generator, discriminator=discriminator,
                                                lr_decay_schedule_discriminator = lr_decay_schedule_discriminator,
                                                lr_decay_schedule_generator = lr_decay_schedule_generator,
                                                **vars(args))

            args.start_epoch = args.start_epoch + stage * args.number_of_epochs

            trainer = Trainer(dataset, gan, at_store_checkpoint_hook=hook,**vars(args))
            trainer.train()
        else:
            hook(0)

        if args.progressive:
            if not os.path.exists(args.tmp_progresive_checkpoints_dir):
                os.makedirs(args.tmp_progresive_checkpoints_dir)
            generator_checkpoint = os.path.join(args.tmp_progresive_checkpoints_dir, 'generator.' + str(stage))
            discriminator_checkpoint = os.path.join(args.tmp_progresive_checkpoints_dir, 'discirminator.' + str(stage))

            generator.save_weights(generator_checkpoint)
            discriminator.save_weights(discriminator_checkpoint)

        return generator_checkpoint, discriminator_checkpoint

    generator_checkpoint = args.generator_checkpoint
    discriminator_checkpoint = args.discriminator_checkpoint

    if args.progressive:
        for stage in range(args.progressive_stage, args.number_of_stages):
            generator_checkpoint, discriminator_checkpoint = run_stage(stage, generator_checkpoint, discriminator_checkpoint)
    else:
        run_stage(0, generator_checkpoint, discriminator_checkpoint)


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
    params.output_channels = 1 if args.dataset == 'mnist' else 3
    params.input_cls_shape = (1, )
    params.block_sizes = tuple([args.generator_filters] * 2) if args.dataset == 'mnist' else tuple([args.generator_filters] * 3)
    params.first_block_shape = (7, 7, args.generator_first_filters) if args.dataset == 'mnist' else (4, 4, args.generator_first_filters)
    params.number_of_classes = 10
    params.concat_cls = args.generator_concat_cls
    params.conditional_bottleneck = 'c' in args.generator_bottleneck
    params.unconditional_bottleneck = 'u' in args.generator_bottleneck
    params.conditional_shortcut = 'c' in args.generator_shortcut
    params.unconditional_shortcut = 'u' in args.generator_shortcut
    params.norm = args.generator_bn != 'n'
    params.conditional_bn = args.generator_bn == 'c'

    params.depthwise = args.generator_depthwise

    params.progressive = args.progressive
    params.progressive_stage = args.progressive_stage
    params.progressive_iters_per_stage = args.number_of_epochs * 1000
    return params


def get_discriminator_params(args):
    params = Namespace()
    params.input_image_shape = args.image_shape
    params.input_cls_shape = (1, )
    params.block_sizes = tuple([args.discriminator_filters] * 4)
    params.resamples = ('DOWN', "DOWN", "SAME", "SAME")
    params.number_of_classes=10
    params.norm = args.discriminator_bn != 'n'
    params.conditional_bn = args.discriminator_bn == 'c'

    params.spectral = args.spectral
    params.fully_diff_spectral = args.fully_diff_spectral
    params.spectral_iterations = args.spectral_iterations
    params.conv_singular = args.conv_singular
    params.renorm_for_cond_singular = args.renorm_for_cond_singular

    params.type = args.gan_type
    params.conditional_bottleneck = 'c' in args.discriminator_bottleneck
    params.unconditional_bottleneck = 'u' in args.discriminator_bottleneck
    params.conditional_shortcut = 'c' in args.discriminator_shortcut
    params.unconditional_shortcut = 'u' in args.discriminator_shortcut

    params.progressive = args.progressive
    params.progressive_stage = args.progressive_stage
    params.progressive_iters_per_stage = args.number_of_epochs * 1000
    
    params.sum_pool = args.sum_pool
    params.depthwise = args.discriminator_depthwise

    return params


def main():
    parser = parser_with_default_args()
    parser.add_argument("--phase", choices=['train', 'test'], default='train')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    parser.add_argument("--lr", default=5e-4, type=float, help="Learning rate")
    parser.add_argument("--beta1", default=0, type=float, help='Adam parameter')
    parser.add_argument("--beta2", default=0.9, type=float, help='Adam parameter')
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'cifar10'], help='Dataset to train on')

    parser.add_argument("--spectral", default=0, type=int, help='Use spectral norm in discriminator')
    parser.add_argument("--fully_diff_spectral", default=0, type=int, help='Fully difirentiable spectral normalization')
    parser.add_argument("--spectral_iterations", default=1, type=int, help='Number of iteration per spectral update')
    parser.add_argument("--conv_singular", default=0, type=int, help='Singular convolution layer')

    parser.add_argument("--generator_bn", default='u', choices=['c', 'u', 'n'],
                        help='Batch nromalization in generator. c - conditional, u - unconditional, n - none')
    parser.add_argument("--generator_concat_cls", default=0, type=int, help='Concat labels to noise in genrator')
    parser.add_argument("--generator_bottleneck", default='no', choices=['c', 'u', 'uc', 'cu', 'no'],
                        help='Bottleneck to use in generator u - unconditional.'
                             'c - conditional, uc - conditional and unconitional'
                             'no - not use bottlenecks')
    parser.add_argument("--generator_shortcut", default='u', choices=['c', 'u', 'uc', 'cu'],
                        help='Shortcut to use in generator u - unconditional. '
                             'c - conditional, uc - conditional and unconitional')
    parser.add_argument("--generator_filters", default=256, type=int,help='Number of filters in generator_block')
    parser.add_argument("--generator_first_filters", default=256,
                        type=int, help='Number of filters in first generator_block')
    parser.add_argument("--generator_depthwise", default=0, type=int, help="Use condtional separable conv in generator")


    parser.add_argument("--gan_type", default=None, choices=[None, 'AC_GAN', 'PROJECTIVE', 'CLS'],
                        help='Type of gan to use. None for unsuperwised.')

    parser.add_argument("--discriminator_bottleneck", default='no', choices=['c', 'u', 'uc', 'cu', 'no'],
                        help='Bottleneck to use in discriminator u - unconditional.'
                             'c - conditional, uc - conditional and unconitional'
                             'no - not use bottlenecks')
    parser.add_argument("--discriminator_shortcut", default='u', choices=['c', 'u', 'uc', 'cu'],
                        help='Shortcut to use in discriminator u - unconditional. '
                             'c - conditional, uc - conditional and unconitional')
    parser.add_argument("--discriminator_bn", default='n', choices=['c', 'u', 'n'],
                        help='Batch nromalization in discriminator. c - conditional, u - unconditional, n - none')
    parser.add_argument("--discriminator_filters", default=128, type=int, help='Number of filters in discriminator_block')
    parser.add_argument("--discriminator_depthwise", default=0, type=int, help="Use condtional separable conv in generator")

    parser.add_argument("--progressive", default=0, type=int, help='Progresive Growing. In progresive mod run number_of_epochs epochs per each stage.')
    parser.add_argument("--tmp_progresive_checkpoints_dir", default='tmp',
                        help='Folder for intermediate checkpoints for progresive')
    parser.add_argument("--progressive_stage", default=0, type=int, help='Stage of progressive growing. 0 if train from scratch.')
    parser.add_argument("--compute_inception", default=1, type=int, help='Compute inception score')
    parser.add_argument("--compute_fid", default=1, type=int, help="Compute fid score")
    parser.add_argument("--plot_model", default=0, type=int)
    parser.add_argument("--print_summary", default=1, type=int, help="Print summary of models")
    parser.add_argument("--lr_decay_schedule", default=None, choices=[None, 'linear', 'half-linear', 'linear-end'],
                        help='Learnign rate decay schedule. None - no decay. '
                             'linear - linear decay to zero. half-linear - linear decay to 0.5'
                             'linear-end constant until 0.9, then linear decay to 0')
    parser.add_argument("--sum_pool", default=1, type=int,
                        help='Use sum or average pooling')
    parser.add_argument("--conditional_optimizer", type=int, default=0,
                        help="Increase lerning rate for conditional layers")
    parser.add_argument("--renorm_for_cond_singular", type=int, default=0,
                        help='If compute one sigma per conditional filter. Otherwise compute number_of_classes sigma.')
    parser.add_argument("--samples_for_evaluation", type=int, default=50000, help='Number of samples for evaluation')

    parser.add_argument("--depthwise", type=int, default=1, help='DepthwiseGenerator')

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

    args.number_of_stages = 5 if args.dataset == 'mnist' else 7
    args.image_shape = (28, 28, 1) if args.dataset == 'mnist' else (32, 32, 3)
    args.fid_cache_file = "output/%s_fid.npz" % args.dataset

    discriminator_params = get_discriminator_params(args)
    generator_params = get_generator_params(args)

    del args.dataset

    compile_and_run(dataset, args, generator_params, discriminator_params)


if __name__ == "__main__":
    main()
