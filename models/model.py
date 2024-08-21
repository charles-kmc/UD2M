import torch #type:ignore
import torch.nn as nn #type:ignore
import torch.nn.utils.parametrize as P#type:ignore
import torch.nn.functional as F#type:ignore
from guided_diffusion.fp16_util import convert_module_to_f16, convert_module_to_f32
from guided_diffusion.nn import (
    checkpoint,
    conv_nd,
    avg_pool_nd,
    zero_module,
    normalization,
    normalization_group,
    timestep_embedding,
    linear,
)
from guided_diffusion.unet import UNetModel, AttentionBlock
from degradation_model.utils_deblurs import blur2operator
from utils.utils import image_transform, inverse_image_transform

import copy
class SequentialBloack_con(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """
    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)



class ResBlock_new(nn.Module):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        dropout,
        norm_grp = 16,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.norm_grp = norm_grp,
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization_group(norm_grp, channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        
        self.out_layers = nn.Sequential(
            normalization_group(norm_grp, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
    
    def forward(self, x):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :return: an [N x C x ...] Tensor of outputs.
        """
        
        return checkpoint(
            self._forward, (x,), self.parameters(), False
        )

    def _forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        
        return self.skip_connection(x) + h





class FineTuningDiffusionModel(nn.Module):
    """
    The full UNet model.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        device,
        frozen_model = None,
        input_block_connection =[8],
        output_block_connection =[1,2],
        norm_grp = 32,
        num_res_blocks = 1,
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=True,
    ):
        super().__init__()
        self.output_block_connection = output_block_connection
        self.input_block_connection = input_block_connection
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.norm_grp = norm_grp
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.device = device

        time_embed_dim = model_channels * 4
        self.time_emb = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
        ch = input_ch = int(channel_mult[0] * model_channels)
        
        self.input_blocks_new = nn.ModuleList(SequentialBloack_con(conv_nd(dims, in_channels, ch, 3, padding=1)))
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock_new(
                        ch,
                        dropout,
                        norm_grp = norm_grp,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                self.input_blocks_new.append(SequentialBloack_con(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                out_ch = ch

                self.input_blocks_new.append(SequentialBloack_con(
                        ResBlock_new(
                            ch,
                            dropout,
                            norm_grp = norm_grp,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch
        
        self.middle_block_new = SequentialBloack_con(
                                ResBlock_new(
                                    ch,
                                    dropout,
                                    norm_grp = norm_grp,
                                    dims=dims,
                                    use_scale_shift_norm=use_scale_shift_norm
                                ),
                            )   
        self._feature_size += ch
        
        # zero convolution
        self.zero_convolution = nn.Conv2d(512, 1024, kernel_size=1, padding = 0, stride=1)
        nn.init.zeros_(self.zero_convolution.weight)                     
            
        if frozen_model is not None:
            self.model = frozen_model
        else:
            raise "You need to provide the required pretrained model"
        
        # Frozen time embedding
        # self.frozen_time_emb = nn.ModuleList(self.model.time_embed)
        
        # Frozen input blocks
        self.frozen_input_blocks  = nn.ModuleList()
        for in_block in self.model.input_blocks.children():
            self.frozen_input_blocks.append(in_block)
        
        # Frozen middle block
        self.frozen_middle_block = self.model.middle_block
        
        # Frozen output blocks
        self.frozen_output_blocks = nn.ModuleList()
        for out_block in self.model.output_blocks.children():
            self.frozen_output_blocks.append(out_block)
        
        # Frozen output layer
        self.frozen_out = self.model.out
        
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks_new.apply(convert_module_to_f16)
        self.middle_block_new.apply(convert_module_to_f16)
        self.zero_convolution.apply(convert_module_to_f16)
        self.frozen_input_blocks.apply(convert_module_to_f16)
        self.frozen_middle_block.apply(convert_module_to_f16)
        self.frozen_output_blocks.apply(convert_module_to_f16)
        self.frozen_out.apply(convert_module_to_f16)
        self.time_emb.apply(convert_module_to_f16)
        # self.frozen_time_emb.apply(convert_module_to_f16)
    
    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks_new.apply(convert_module_to_f32)
        self.middle_block_new.apply(convert_module_to_f32)
        self.zero_convolution.apply(convert_module_to_f32)
        self.frozen_input_blocks.apply(convert_module_to_f32)
        self.frozen_middle_block.apply(convert_module_to_f32)
        self.frozen_output_blocks.apply(convert_module_to_f32)
        self.frozen_out.apply(convert_module_to_f32)
        self.time_emb.apply(convert_module_to_f32)
        # self.frozen_time_emb.apply(convert_module_to_f32)

    def forward(self, x, timesteps = torch.tensor([3]), y = None):
        
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if y is not None:
            h = y.type(self.dtype)
        else:
            h = x.type(self.dtype)
        x = x.type(self.dtype)
        h4concatenation = []
        h4zero_convolution = []
        xs4skip_connection = []
        
        # time embedding
        emb = timestep_embedding(timesteps, self.model_channels)
        emb = self.time_emb(emb)
        
        # Input blocks
        for ii, module in enumerate(self.input_blocks_new):
            h = module(h)
            if ii in self.input_block_connection:
                h4concatenation.append(h)
            h4zero_convolution.append(h)
        for ii, frozen_module in enumerate(self.frozen_input_blocks):
            x = frozen_module(x, emb)
            if ii in self.input_block_connection:
                x = x + h4concatenation.pop(0)
            xs4skip_connection.append(x) 

        # Middle block with conditioning
        h = self.middle_block_new(h)
        x = x + h 
        x = self.frozen_middle_block(x, emb)
        
        # output blocks
        for ii, frozen_module in enumerate(self.frozen_output_blocks):
            h = h4zero_convolution.pop() 
            x = torch.cat([x, xs4skip_connection.pop()], dim=1)
            if ii == 1: # or ii == 2:
                h = self.zero_convolution(h)
                x = x + h 
            x = frozen_module(x, emb)
        
        # output layer    
        x = x.type(x.dtype)
        x = self.frozen_out(x)          
        return x
    
    # -- Consistency loss
    def consistency_loss(self, xpred, y, list_op_blur):
        
        mse_function = nn.MSELoss()
        device = xpred.device
        
        # Create a list of kernels and images
        kernels = list_op_blur["blur_value"] 
        sigmas = list_op_blur["sigma"].to(device)
        mse_per_sample = []
        
        # Loop over kernels and xpreds in parallel
        for i in range(kernels.shape[0]):
            kernel = kernels[i,...]
            xpred_i = inverse_image_transform(xpred[i,...])
            xpred_fft = torch.fft.fft2(xpred_i, dim=(-2, -1))
            FFT_H = blur2operator(kernel, self.image_size).to(device)
            blurred_fft = FFT_H * xpred_fft
            ax_pred = torch.real(torch.fft.ifft2(blurred_fft, dim=(-2, -1)))
            mse_per_sample.append(mse_function(ax_pred, y[i]).item())
        
        mse_val = torch.tensor(mse_per_sample).to(device)
        loss =  mse_val / (2*sigmas**2)
        
        return loss.mean()


                
                
           
