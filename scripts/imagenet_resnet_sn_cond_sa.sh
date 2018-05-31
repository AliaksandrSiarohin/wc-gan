#!/usr/bin/env bash
python run.py --dataset imagenet --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm ufconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 1024 --generator_filters 512\
 --discriminator_spectral 1 --lr_decay_schedule linear-end --number_of_epochs 450 --gan_type PROJECTIVE --filters_emb 32
