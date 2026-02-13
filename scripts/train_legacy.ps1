# Default
# python train_word_level.py `
#     --batch_size 8 `
#     --epochs 1000 `
#     --rec_start_epoch 50 `
#     --rec_weight_start 0.001 `
#     --rec_weight_max 0.05 `
#     --rec_curriculum_epochs 150 `
#     --wandb_log True `
#     --train_mode train `
#     --load_check True `

# current ly we started after 93 epochs
python train_word_level.py `
    --batch_size 8 `
    --epochs 1000 `
    --rec_start_epoch 1 `
    --rec_weight_start 0.03170666666666667 `
    --rec_weight_max 0.05 `
    --rec_curriculum_epochs 57 `
    --wandb_log True `
    --train_mode train `
    --load_check True `
    --num_workers 4 `
    
    


