import argparse

def configs(mode="train"):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config_file", type=str, required=True, help="configs file")
    parser.add_argument("--task", type=str, required=True, help="Task we are solving")
    parser.add_argument("--operator_name", type=str, required=True, help="Type of operator used!!")
    parser.add_argument("--max_unfolded_iter", type=int, default=3, help="Type of operator used!!")
    parser.add_argument("--kernel_size", type=int, default=3, help="Kernel size!!")
    parser.add_argument("--learned_lambda_sig", type=bool, default=True, help="Whether to learn lambda and sigma or not!!")
    parser.add_argument("--lambda_", type=float, required=True, help="parameter controlling data consistency step!!")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name!!")
    parser.add_argument("--sigma_model", type=float, required=True, help="Noise variance of the model")
    parser.add_argument("--use_RAM", type=bool, default=False, help="Whether to use RAM or not")

    if mode!="train":
        parser.add_argument("--num_timesteps", type=int, required=True, help="Number of timesteps for inference")
        parser.add_argument("--max_images", type=int, required=True, help="Maximum number of images to process")
        parser.add_argument("--init_prev", action="store_true", help="Use the previous estimate as initialization")
        parser.add_argument("--ckpt_name", type=str, default=None, help="checkpoint name")
        parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory")
        parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per image")
        parser.add_argument("--equivariant", action="store_true", help="Use equivariant architecture")
    config = parser.parse_args()
    
    if config.task == "inp":
        config.use_dpir = False
    return config

