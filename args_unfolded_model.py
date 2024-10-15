
from utils.utils import DotDict
def args_unfolded():
    main_args = {
        "task":"debur",
        "resume_epoch":0,
        "resume_model":0,
        "learning_rate":3e-5,
        "weight_decay":1e-2,
        "rank":10,
        "seed":46,
        "save_checkpoint_range":10,
        "use_consistancy":False,
        "use_wandb":True,
        "save_progressive":True,
        "init_xt_y":False,
        "pertub":True,
        "train_batch_size":8,
        "shuffle":True,
        "test_batch_size":1,
        "num_workers":0,
        "dataset_name":"FFHQ",
        "im_size":256,
        "mode":"train",
        "dataset_path":"/users/cmk2000/sharedscratch/Datasets/ffhq_dataset",
        "save_checkpoint_dir":"/users/cmk2000/sharedscratch/Pretrained-Checkpoints/Conditional-diffusion_model_for_ivp",
        # "save_checkpoint_dir":"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model_for_ivp/Checkpoints",
        "path_save": "/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp"
    }
    pyhsic = {
        "kernel_size": 25,
        "blur_name":"gaussian",
        "sigma_model":0.05,
        "transform_y": True,
    }
    dpir = {
        "pretrained_pth": "/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo",
        "model_name": "drunet_color", #"ircnn_color, drunet_color",
        "use_dpir":True,
    }
    evaluation = {
        "fid":True,
        "swd":True,
        "cmmd":True,
        "metrics":True,
        "skipper":2,
        "coverage":True,
    }
    args = DotDict(main_args)
    args.physic = DotDict(pyhsic)
    args.dpir = DotDict(dpir)
    args.evaluation = DotDict(evaluation)
    return args