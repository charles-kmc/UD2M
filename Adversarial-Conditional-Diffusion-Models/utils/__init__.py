from .utils import (
    DotDict,
    sigma_eval,
    projbox,
    welford,
    Metrics,
    image_transform,
    inverse_image_transform,
    save_images,
    tensor2uint,
    delete,
    get_batch_rgb_from_batch_tensor,
    get_rgb_from_tensor
)
from .get_logger import get_loggers
from .lora_parametrisation import LoRa_model,enable_disable_lora
from .fftc import fft2c_new, ifft2c_new

from .math import tensor_to_complex_np, complex_mul,complex_conj,complex_abs,complex_abs_sq
from types import SimpleNamespace

class namespace(dict):
    """Enable dot notation access to dictionary attributes."""
    def __getattr__(self, attr):
        value = self.get(attr)
        if isinstance(value, dict):
            return namespace(value)
        return value
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# transforming a dictionary to a dot dictionary   
def dict_to_dotdict(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_dotdict(v) for k, v in d.items()})
    return d