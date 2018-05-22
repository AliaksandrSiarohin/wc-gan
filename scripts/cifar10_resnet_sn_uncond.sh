#!/usr/bin/env bash
python run.py --dataset cifar10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --lr 2e-4 --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 256\
 --spectral 1 --lr_decay_schedule linear --number_of_epochs 50
