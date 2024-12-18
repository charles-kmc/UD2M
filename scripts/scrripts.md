## Training scripts

- Deblurring
    - sbatch ./scripts/tdeblur_unfolded_gaussian.sh
    - python3 main_sample_conditional.py --task "deblur" --operator_name "gaussian" --lambda_ 0.3 --ckpt_epoch 990 --ckpt_date "17-11-2024"

    - sbatch ./scripts/tdeblur_unfolded_uniform_motion.sh
    - python3 main_sample_conditional.py --task "deblur" --operator_name "uniform_motion" --lambda_ 0.7 --ckpt_epoch 90 --ckpt_date "13-12-2024"

    - sbatch ./scripts/tdeblur_unfolded_motion.sh
    - python3 main_sample_conditional.py --task "deblur" --operator_name "motion" --lambda_ 0.3 --ckpt_epoch 300 --ckpt_date "07-12-2024"
- Inpainting
    - sbatch ./scripts/tinp_unfolded_random.sh
    - python3 main_sample_conditional.py --task "inp" --operator_name "random" --lambda_ 1000 --ckpt_epoch 300 --ckpt_date "07-12-2024"
