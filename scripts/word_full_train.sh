python train_word_level.py \
    --batch_size 8 \
    --epochs 1000 \
    --rec_start_epoch 50 \
    --rec_weight_start 0.001 \
    --rec_weight_max 0.05 \
    --rec_curriculum_epochs 150 --wandb_log True --train_mode train --load_check True


