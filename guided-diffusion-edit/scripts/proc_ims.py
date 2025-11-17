from argparse import ArgumentParser
from guided_diffusion.script_util import add_dict_to_argparser

def create_argparser():
    defaults = dict(
        im_file="samples.npz",
    )
    parser = ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def main():
    args = create_argparser().parse_args()
    import numpy as np
    ims = np.load(args.im_file)["arr_0"]

    import matplotlib.pyplot as plt
    num_ims = ims.shape[0]
    rt_ax = int(np.sqrt(num_ims))
    fig, axes = plt.subplots(rt_ax, rt_ax, figsize=(8, 8))
    for i in range(rt_ax):
        for j in range(rt_ax):
            axes[i, j].imshow(ims[i * rt_ax + j], cmap="gray")
            axes[i, j].axis("off")
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(args.im_file.replace(".npz", ".png"))
    plt.close()

if __name__ == "__main__":
    main()