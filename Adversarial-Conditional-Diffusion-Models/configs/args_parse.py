import argparse

def configs(mode="train"):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--config_file", type=str, required=True, help="configs file")
    parser.add_argument("--task", type=str, required=True, help="Task we are solving")
    parser.add_argument("--operator_name", type=str, required=True, help="Type of operator used!!")
    parser.add_argument("--lambda_", type=float, required=True, help="parameter controlling data consistency step!!")
    parser.add_argument("--dataset", type=str, default="FFHQ", help="Dataset!!")
    if mode == "inference":
        parser.add_argument("--ckpt_epoch", type=int, required=True, help="Epoch where the model is resumed")
        parser.add_argument("--sigma_model", type=float, required=True, help="Noise variance of the model")
        parser.add_argument("--ckpt_date", type=str, required=True, help="Date when the checkpoint was save")
        parser.add_argument("--total_steps", type=int, required=True, help="Total timesteps")
    config = parser.parse_args()
    
    if config.task == "inp":
        config.use_dpir = False
    return config



