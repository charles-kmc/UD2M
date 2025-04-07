#### Results

|Dataset | Operation | Noise | Comparisons | Results |
|---| ---| ---| --- | --- |
|FFHQ (256x256) | Gaussian Deblurring | $\mathcal{N}(0, \sigma_y^2)$ | Ours | <span style = "color:orange"> In progress </span> |
| | Motion Deblurring || DiffPIR | Available at [DiffPIR](https://arxiv.org/pdf/2305.08995) |
| | Super-Resolution |  | DDRM | Available at [DiffPIR](https://arxiv.org/pdf/2305.08995) | 
| | Box-Inpainting |  | DPS  | Available at [DiffPIR](https://arxiv.org/pdf/2305.08995)|
| | | | DPIR |Available at [DiffPIR](https://arxiv.org/pdf/2305.08995) |
| | | | <span style="color:red">Additional Methods which require training for Inverse Problem.</span> | <span style="color:red">TBC</span> |
|LSUN Bedroom (256x256) | Box-Inpainting (128x128 centre box) | $\mathcal N(0, 0.05^2)$ | DDRM | Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | Consistency Model| Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | DPS|Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | |[I $^2$ SB](https://i2sb.github.io/)| Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | [CDDB](https://arxiv.org/abs/2305.19809) |Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | [CoSIGN](https://arxiv.org/abs/2407.12676)|Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | [CM4IR](https://arxiv.org/html/2412.20596v2)| <span style = "color:red">TBC</span>, [code](https://github.com/tirer-lab/CM4IR)|
| | | | DiffPIR | <span style = "color:red"> TBC </span> (Pretrained DDPM at [code](https://github.com/pesser/pytorch_diffusion), [weights](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/))|
| | | | Ours | <span style = "color:red">TBC</span> (Pretrained DDPM at [code](https://github.com/pesser/pytorch_diffusion), [weights](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/))|
||||
|LSUN Bedroom (256x256) | 4x Super-Resolution | $\mathcal N(0, 0.05^2)$ | DDRM |Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | Consistency Model|Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | DPS|Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | |[I $^2$ SB](https://i2sb.github.io/)|Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | [CDDB](https://arxiv.org/abs/2305.19809) |Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | [CoSIGN](https://arxiv.org/abs/2407.12676)|Available through [CoSIGN](https://arxiv.org/abs/2407.12676)|
| | | | [CM4IR](https://arxiv.org/html/2412.20596v2)|Available through [CM4IR](https://arxiv.org/html/2412.20596v2)|
| | | |DiffPIR| Available through [CM4IR](https://arxiv.org/html/2412.20596v2)
| | | | Ours | <span style = "color:red"> TBC </span> (Pretrained DDPM at [code](https://github.com/pesser/pytorch_diffusion), [weights](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/))
||||
|LSUN Bedroom (256x256) | Gaussian Deblurring (From the code [here, L231](https://github.com/tirer-lab/CM4IR/blob/main/guided_diffusion/diffusion.py) the authors use an anisotropic Gaussian kernel with st. dev. 1 and 20 in each axis, respectively) | $\mathcal N(0, 0.05^2)$ | DDRM |Available through [CM4IR](https://arxiv.org/html/2412.20596v2)|
| |Random Inpainting (80% pixels missing) | |  Consistency Model|Available through [CM4IR](https://arxiv.org/html/2412.20596v2)|
| | | | DPS|<span style="color:red">TBC</span>|
| | | |[I $^2$ SB](https://i2sb.github.io/)|<span style="color:red">TBC</span>|
| | | | [CDDB](https://arxiv.org/abs/2305.19809) |<span style="color:red">TBC</span>|
| | | | [CoSIGN](https://arxiv.org/abs/2407.12676)|Available through [CM4IR](https://arxiv.org/html/2412.20596v2)|
| | | | [CM4IR](https://arxiv.org/html/2412.20596v2)|Available through [CM4IR](https://arxiv.org/html/2412.20596v2)|
| | | |DiffPIR| Available through [CM4IR](https://arxiv.org/html/2412.20596v2)
| | | | Ours | <span style = "color:red"> TBC </span> (Pretrained DDPM at [code](https://github.com/pesser/pytorch_diffusion), [weights](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/))


Notes: 

- In The [CM4IR](https://arxiv.org/html/2412.20596v2) paper, the authors state that the CoSIGN code first maps images to 
the range [-1,1] before adding the $\mathcal N(0, 0.05^2)$ noise. Hence, the true noise level is 0.025.

- I have a fork of the Conditional GAN [repository](https://github.com/matt-bendel/rcGAN), it would not be too difficult to train a GAN for each of the above tasks, to compare to a non-diffusion based method. 
