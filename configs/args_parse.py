import argparse

def configs(mode="train"):
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--task", type=str, required=True, help="Task we are solving")
    parser.add_argument("--operator_name", type=str, required=True, help="Type of operator used!!")
    if mode == "inference":
        parser.add_argument("--ckpt_epoch", type=int, required=True, help="Epoch where the model is resumed")
        parser.add_argument("--ckpt_date", type=str, required=True, help="Date when the checkpoint was save")
    config = parser.parse_args()
    return config



