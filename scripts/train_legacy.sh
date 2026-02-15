#!/usr/bin/env bash

python train_word_level.py \
  --batch_size 8 \
  --epochs 1000 \
  --rec_start_epoch 0 \
  --rec_weight_start 0.032669473684210526 \
  --rec_weight_max 0.05 \
  --rec_curriculum_epochs 54 \
  --wandb_log True \
  --train_mode train \
  --load_check True \
  --num_workers 4
