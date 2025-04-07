# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss function."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import RandomCrop

import dnnlib
from torch_utils import training_stats
from torch_utils.ops import upfirdn2d
from WGAN.clip import CLIP


class ProjectedGANLoss:
    def __init__(
        self,
        device: torch.device,
        x0_gen: torch.Tensor,
        D: nn.Module
    ):
        super().__init__()
        self.device = device
        self.x0_gen = x0_gen
        self.D = D

    def accumulate_gradients(
        self,
        real_img: torch.Tensor,
    ) -> None:
        
        batch_size = real_img.size(0)


        # ---> Discriminator
        # Minimize logits for generated images.
        gen_logits = self.D(self.x0_gen)
        loss_gen = (F.relu(torch.ones_like(gen_logits) + gen_logits)).mean() / batch_size
        loss_gen.backward()

        # Maximize logits for real images.
        real_img_tmp = real_img.detach().requires_grad_(False)
        real_logits = self.D(real_img_tmp)
        loss_real = (F.relu(torch.ones_like(real_logits) - real_logits)).mean() / batch_size
        loss_real.backward()


        # ---> Discriminator
        # Maximize logits for generated images.
        gen_logits = self.D(self.x0_gen)
        loss_gen = (-gen_logits).mean() / batch_size

        loss_gen.backward()
