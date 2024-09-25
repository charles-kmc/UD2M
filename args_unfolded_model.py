
from utils.utils import DotDict
def args_unfolded():
    main_args = {
        "task":"debur",
        "learning_rate":3e-6,
        "weight_decay":1e-2,
        "rank":5,
        "use_consistancy":True,
        "use_wandb":True,
        "save_progressive":True,
        "train_batch_size":8,
        "shuffle":True,
        "test_batch_size":1,
        "num_workers":0,
        "dataset_name":"FFHQ",
        "im_size":256,
        "dataset_path":"/users/cmk2000/sharedscratch/Datasets/ffhq_dataset",
        "save_checkpoint_dir":"/users/cmk2000/sharedscratch/Pretrained-Checkpoints/conditional-diffusion_model-for_ivp",
    }
    pyhsic = {
        "kernel_size": 25,
        "blur_name":"gaussian",
        "sigma_model":0.05,
    }
    args = DotDict(main_args)
    args.physic = DotDict(pyhsic)
    return args