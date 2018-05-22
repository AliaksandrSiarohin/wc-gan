#!/usr/bin/env bash
python run.py --dataset cifar10 --generator_adversarial_objective wgan\
 --discriminator_adversarial_objective wgan --lr 2e-4 --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 128\
 --gradinet_penalty_weight 10 --lr_decay_schedule linear --number_of_epochs 100
