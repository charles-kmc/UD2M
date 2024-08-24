from inference.inference_diffpir import inference_deb

noise_vals = [0.01]
blur_modes = ["gaussian"]  # "gaussian" | "motion"

# deblurring inference
for noise_val in noise_vals:
    for blur_mode in blur_modes:
        inference_deb(noise_val, blur_mode)



