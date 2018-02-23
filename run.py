from keras.optimizers import Adam

from gan.dataset import LabeledArrayDataset
from gan.cmd import parser_with_default_args
from gan.train import Trainer
from gan.ac_gan import AC_GAN
from gan.projective_gan import ProjectiveGAN
from gan.gan import GAN

import os
import json
from functools import partial
from scorer import compute_scores
from time import time
from argparse import Namespace

from generator import make_generator
from discriminator import make_discriminator
from keras.utils import plot_model


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

    args.generator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)
    args.discriminator_optimizer = Adam(args.lr, beta_1=args.beta1, beta_2=args.beta2)

    log_file = os.path.join(args.output_dir, 'log.txt')

    at_store_checkpoint_hook = partial(compute_scores, image_shape=args.image_shape, log_file=log_file,
                                       dataset=dataset, compute_inception=args.compute_inception,
                                       compute_fid=args.compute_fid, additional_info=additional_info)

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
            GANS = {None:GAN, 'AC_GAN':AC_GAN, 'PROJECTIVE':ProjectiveGAN}
            gan = GANS[args.discriminator_type](generator=generator, discriminator=discriminator, **vars(args))

            args.start_epoch = args.start_epoch + stage * args.number_of_epochs

            trainer = Trainer(dataset, gan, at_store_checkpoint_hook=hook,**vars(args))
            trainer.train()
        else:
            at_store_checkpoint_hook()

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


def get_generator_params(args):
    params = Namespace()
    params.output_channels = 1 if args.dataset == 'mnist' else 3
    params.input_cls_shape = (1, )
    params.block_sizes = (128, 128) if args.dataset == 'mnist' else (256, 256, 256)
    params.first_block_shape = (7, 7, 256) if args.dataset == 'mnist' else (4, 4, 256)
    params.number_of_classes = 10
    params.concat_cls = (args.generator_type == "CONCAT")
    params.conditional_bottleneck = (args.generator_type == "BOTTLENECK")
    params.unconditional_bottleneck = False
    params.conditional_shortcut = (args.generator_type == "SHORTCUT")
    params.unconditional_shortcut = (args.generator_type != "SHORTCUT")
    params.conditional_bn = (args.generator_type == "COND_BN")

    params.progressive = args.progressive
    params.progressive_stage = args.progressive_stage
    params.progressive_iters_per_stage = args.number_of_epochs * 1000
    return params


def get_discriminator_params(args):
    params = Namespace()
    params.input_image_shape = args.image_shape
    params.input_cls_shape = (1, )
    params.block_sizes = (128, 128, 128, 128)
    params.resamples = ('DOWN', "DOWN", "SAME", "SAME")
    params.number_of_classes=10
    params.norm = args.bn_in_discriminator

    params.spectral = args.spectral
    params.fully_diff_spectral = args.fully_diff_spectral
    params.spectral_iterations = args.spectral_iterations
    params.conv_singular = args.conv_singular

    params.type = args.discriminator_type
    params.conditional_bottleneck = (args.discriminator_type == "BOTTLENECK")
    params.unconditional_bottleneck = False
    params.conditional_shortcut = (args.discriminator_type == "SHORTCUT")
    params.unconditional_shortcut = (args.discriminator_type != "SHORTCUT")

    params.progressive = args.progressive
    params.progressive_stage = args.progressive_stage
    params.progressive_iters_per_stage = args.number_of_epochs * 1000
    return params


def main():
    parser = parser_with_default_args()
    parser.add_argument("--phase", choices=['train', 'test'], default='train')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    parser.add_argument("--lr", default=2e-4, type=float, help="Learning rate")
    parser.add_argument("--beta1", default=0, type=float, help='Adam parameter')
    parser.add_argument("--beta2", default=0.9, type=float, help='Adam parameter')
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'cifar10'], help='Dataset to train on')

    parser.add_argument("--spectral", default=0, type=int, help='Use spectral norm in discriminator')
    parser.add_argument("--fully_diff_spectral", default=0, type=int, help='Fully difirentiable spectral normalization')
    parser.add_argument("--spectral_iterations", default=1, type=int, help='Number of iteration per spectral update')
    parser.add_argument("--conv_singular", default=0, type=int, help='Singular convolution layer')

    parser.add_argument("--generator_type", default=None, choices=[None, "CONCAT", "COND_BN", "BOTTLENECK", "SHORTCUT"],
                        help='Type of generator to use. None for unsuperwised')
    parser.add_argument("--discriminator_type", default=None, choices=[None, 'AC_GAN', 'PROJECTIVE', 'BOTTLENECK', 'SHORTCUT'],
                        help='Type of generator to use. None for unsuperwised')

    parser.add_argument("--uncoditional_bottleneck", default=0, type=int,
                        help='Use uncoditional conv11 layer in the middle of resblock')
    parser.add_argument("--bn_in_discriminator", default=0, type=int, help='Use batch nromalization in discriminator')
    parser.add_argument("--progressive", default=0, type=int, help='Progresive Growing. In progresive mod run number_of_epochs epochs per each stage.')
    parser.add_argument("--tmp_progresive_checkpoints_dir", default='tmp',
                        help='Folder for intermediate checkpoints for progresive')
    parser.add_argument("--progressive_stage", default=0, type=int, help='Stage of progressive growing. 0 if train from scratch.')
    parser.add_argument("--compute_inception", default=1, type=int, help='Compute inception score')
    parser.add_argument("--compute_fid", default=1, type=int, help="Compute fid score")
    parser.add_argument("--plot_model", default=0, type=int)
    parser.add_argument("--print_summary", default=1, type=int, help="Print summary of models")


    args = parser.parse_args()

    dataset = get_dataset(dataset=args.dataset,
                          batch_size=args.batch_size,
                          supervised=args.generator_type is not None)

    args.output_dir = "output/%s_%s" % (args.dataset, time())
    args.checkpoints_dir = args.output_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as outfile:
        json.dump(vars(args), outfile, indent=4)

    args.number_of_stages = 5 if args.dataset == 'mnist' else 7
    args.image_shape = (28, 28, 1) if args.dataset == 'mnist' else (32, 32, 3)


    discriminator_params = get_discriminator_params(args)
    generator_params = get_generator_params(args)

    del args.dataset

    compile_and_run(dataset, args, generator_params, discriminator_params)


if __name__ == "__main__":
    main()
