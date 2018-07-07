#!/usr/bin/env bash
name=$(basename $0)
name=${name%.*}
python run.py --name $name --dataset cifar10 --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 512 --generator_filters 512\
 --discriminator_spectral 1 --lr_decay_schedule linear --number_of_epochs 100 --arc dcgan --training_ration 1 --generator_batch_multiple 1
