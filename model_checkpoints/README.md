# Model Checkpoints

This directory contains trained checkpoints for UD2M models used in the numerical experiments presented in the paper.

## Pre-trained SBM weights
| Dataset      | Link                 |
|--------------|-----------------------|
|MNIST         | [Download](https://drive.google.com/file/d/1xXAN4W1sadeEpDwaJCeDESYdiNkj8FV8/view?usp=drive_link)|
|ImageNet      | [Download](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh)   |
|LSUN Bedroom  | [Download](https://heibox.uni-heidelberg.de/d/01207c3f6b8441779abf/)                                                                                        |

## Available Checkpoints

| Dataset      | Task                  | Description | Checkpoint Link                          |
|--------------|-----------------------|-------------|----------------------------|
|MNIST         | SR(x8)                | $K=1$       |[Download](model_checkpoints/MNIST_SR_1.ckpt) |
|         |               | $K=2$       |[Download](model_checkpoints/MNIST_SR_2.ckpt) |
|        |                 | $K=4$       |[Download](model_checkpoints/MNIST_SR_4.ckpt) |
|         |                 | $K=8$       |[Download](model_checkpoints/MNIST_SR_8.ckpt) |
|||||
|LSUN Bedroom | Gaussian Deblurring | $K=3$ with RAM | [Link](https://drive.google.com/file/d/1l7k_keC45SuVIc-we3oX_lGyNYayg81F/view?usp=sharing) |
| | Box Inpainting | $K=3$ with RAM | [Link](https://drive.google.com/file/d/1GUCsKfryIsDNT6IEuy7c-3wp40WIxBGb/view?usp=sharing) |
| | Random Inpainting | $K=3$ with RAM | [Link](https://drive.google.com/file/d/1PECb_9mHgiAatuNsgOFd_SU3parZEqnq/view?usp=sharing) |
| | SR(x4) | $K=3$ with RAM | [Link](https://drive.google.com/file/d/1b88WqVJ0B0IM_lKlVwa7WefjSRgCBP7y/view?usp=sharing) |
|||||
|ImageNet | Gaussian Deblurring | $K=3$ | [Link](https://drive.google.com/file/d/1tbQ76_ryRxSlGLyM3L1sfdzk_wLhHlBx/view?usp=sharing) |
| | Random Inpainting | $K=3$ | [Link](https://drive.google.com/file/d/1O85Jmt8k4bNW8Jp-PtKx6VUcNMMa7F6j/view?usp=sharing) |
| | SR(x4)| $K=3$ | [Link](https://drive.google.com/file/d/1txkWvJ3uEhGfmtO_zXZZH7bCcbaTy55E/view?usp=sharing) |
| | JPEG | $K=3$ | [Link](https://drive.google.com/file/d/1s215fhT_6NMl0PxIOToRjpTQn_dEM2EN/view?usp=sharing) |
