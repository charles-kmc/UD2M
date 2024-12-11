
from utils.utils import DotDict
def args_unfolded():
    main_args = {
        "lambda_":0.1,
        "task":"inp", # inp, deblur
        "noise_schedule_type":"linear",
        "ddpm_param":1,
        "resume_epoch": 0,
        "resume_model": False,
        "learning_rate":3e-5,#3e-5,
        "weight_decay":1e-2,
        "rank":10,
        "seed":46,
        "save_checkpoint_range":30,
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
        # "save_checkpoint_dir":"/users/cmk2000/sharedscratch/Pretrained-Checkpoints/Conditional-diffusion_model_for_ivp",
        "save_checkpoint_dir":"/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model_for_ivp",
        "path_save": "/users/cmk2000/sharedscratch/CKM/Conditional-diffusion_model-for_ivp"
    }
    pyhsic = {
        "kernel_size": 25,
        "operator_name":"gaussian",
        "random_blur": False,
        "sigma_model":0.05,
        "transform_y": True,
    }
    inp = {
        "mask_rate":0.5,
        "operator_name":"random",
        "box_proportion":0.3,
        "index_ii":None,
        "index_jj":None,
    }
    deblur = {
        "random_blur": False,
    }
    
    dpir = {
        "pretrained_pth": "/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo",
        "model_name": "drunet_color", #"ircnn_color, drunet_color",
        "use_dpir":True,
    }
    evaluation = {
        "save_images":True,
        "fid":True,
        "swd":True,
        "cmmd":True,
        "metrics":True,
        "skipper":2,
        "coverage":False,
    }
    model = {
        "model_name":"diffusion_ffhq_10m",
        "pretrained_model_path":"/users/cmk2000/sharedscratch/Pretrained-Checkpoints/model_zoo"
    }
    args = DotDict(main_args)
    args.physic = DotDict(pyhsic)
    args.physic.inp = DotDict(inp) 
    args.physic.deblur = DotDict(deblur) 
    args.dpir = DotDict(dpir)
    args.evaluation = DotDict(evaluation)
    args.model = DotDict(model)
    if args.task == "inp":
        args.dpir.use_dpir = False
        args.lambda_ = 1000
        args.physic.kernel_size = args.im_size
    if args.task == "deblur":
        args.lambda_ = 0.3
    return args