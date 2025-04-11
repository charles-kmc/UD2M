import argparse
from PIL import Image
from pathlib import Path
import os

def rename_images(folder_name, data_dir, out_dir):
    # Set your root directory
    data_dir = Path(data_dir)
    print("get file name ...")
    # Recursively find all .png files
    png_files = list(data_dir.rglob("*.webp"))

    d_th = f"{out_dir}/{folder_name}"
    os.makedirs(d_th, exist_ok=True)

    print(len(png_files))
    
    for jj, webp_path in enumerate(png_files):
        # Open image
        img = img = Image.open(webp_path)
        name = f"{folder_name}_{(jj):07d}"
        
        # save image
        img.save(os.path.join(d_th,f"{name}.png"), "PNG")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', default='/users/cmk2000/sharedscratch/Datasets/LSUN')
    parser.add_argument('-f', '--folder', default="train")
    parser.add_argument('-d', '--data_dir', default="/users/cmk2000/sharedscratch/Datasets/LSUN/data_training")
    args = parser.parse_args()
    
    rename_images(args.folder, args.data_dir, args.out_dir)