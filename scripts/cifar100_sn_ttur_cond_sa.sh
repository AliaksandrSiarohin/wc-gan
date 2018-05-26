#!/usr/bin/env bash
python run.py --dataset cifar100 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --generator_block_norm d --generator_block_after_norm ufconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 256 --generator_filters 128\
 --discriminator_spectral 1 --lr_decay_schedule linear --number_of_epochs 200 --gan_type PROJECTIVE --filters_emb 10\
 --training_ratio 1 --generator_batch_multiple 1 --generator_lr 1e-4 --discriminator_lr 4e-4
