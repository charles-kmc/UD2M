from guided_diffusion import dist_util
from utils import utils_model
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

def load_pre_trained_model(model_name, model_checkpoint_path, device):
    
    # load model
    model_config = dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_checkpoint_path=model_checkpoint_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    
    # agruments for models   
    args = utils_model.create_argparser(model_config).parse_args([])
    
    # create model and diffusion
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_checkpoint_path, map_location="cpu")
    )
    
    # set model to eval mode
    model.eval()
    for _k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    return model, diffusion