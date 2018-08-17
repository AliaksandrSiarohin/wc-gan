#!/usr/bin/env bash
name=$(basename $0)
name=${name%.*}
python run.py --name $name --dataset imagenet --generator_adversarial_objective hinge\
 --discriminator_adversarial_objective hinge  --generator_block_norm d --generator_block_after_norm ufconv\
 --generator_last_norm d --generator_last_after_norm uconv --discriminator_filters 1024 --generator_filters 1024\
 --discriminator_spectral 1 --lr_decay_schedule linear --number_of_epochs 1000 --gan_type PROJECTIVE\
 --filters_emb 32 --batch_size 32 --shred_disc_batch 1
