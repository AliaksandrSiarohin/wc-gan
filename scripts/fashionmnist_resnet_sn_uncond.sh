#!/usr/bin/env bash
name=$(basename $0)
name=${name%.*}
python run.py --name $name --dataset fashion-mnist --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm uconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 128 --generator_filters 256\
 --discriminator_spectral 1 --number_of_epochs 20 --lr_decay_schedule linear --samples_inception 0
