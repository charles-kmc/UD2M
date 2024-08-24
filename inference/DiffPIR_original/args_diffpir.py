
from utils.utils import DotDict
import os

def args_dict():
    diffpir_parameters = dict(
                kernel_size         =  25,
                num_train_timesteps = 1000,
                skip_type           = "quad",
                iter_num            = 100,               
                iter_num_U          = 1,                 
                ddim_sample         = False ,
                model_path          = "/users/cmk2000/sharedscratch/Pretrained-Checkpoints/conditional-diffusion_model-for_ivp/21-08-2024/prop_data_0.1",
                sf                  = 1.0,
                eta                 = 0.0,
                log_process         = False,
                save_LEH            = False,
                save_im             = False,
                save_data           = False,
                calc_LPIPS          = True,
                save_progressive    = True,
                task                = "deblur",
                eta                 = 0.0,
                guidence_scale      = 1,
                testset_name        = "imagenet",
                test_dataset_dir    = "/users/cmk2000/sharedscratch/Datasets/testsets",
                save_data           = 0,
                results_dir         = "/users/cmk2000/sharedscratch/Results/conditional-diffusion_model-for_ivp",
                
            )
    
    args = DotDict(diffpir_parameters)
    args.skip = args.num_train_timesteps//args.iter_num
    args.test_dataset_dir = os.path.join(args.test_dataset_dir, args.testset_name)
    return args