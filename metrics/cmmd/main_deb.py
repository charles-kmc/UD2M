# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main entry point for the CMMD calculation."""
import os
import pandas as pd
from absl import app
from absl import flags
import distance
import embedding
import io_util
import numpy as np
from utils import utils_logger
from logging.handlers import RotatingFileHandler
import logging
import matplotlib.image as mpimg

import cv2
import shutil


_BATCH_SIZE = flags.DEFINE_integer(
    'batch_size', 12, 'Batch size for embedding generation.'
)
_MAX_COUNT = flags.DEFINE_integer(
    'max_count', -1, 'Maximum number of images to read from each directory.'
)



def save_images(dir, image, name):
    image_array = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    cv2.imwrite(os.path.join(dir,name+'.png'), image_array)
 
def rewrite_images(dir):
    dir0, folder = dir.rsplit('/', 1)
    data = os.listdir(dir)

    for i in range(len(data)):
        data[i] = dir +'/' + data[i]
    
    for i_im in range(len(data)): 
        filename = data[i_im]
        dataname = filename.split("/")[-1]
        dataname = dataname.split(".")[0]

        im = mpimg.imread(filename) 
        if im.shape[2] == 3:
            image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        elif im.shape[2] == 4:
            im = im[:,:,:3]
            image_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        else:
          exit()
        
        # Convert RGB to RGBA (add an alpha channel if not present)
        if image_rgb.shape[2] == 3:
            alpha_channel = 255 * np.ones((image_rgb.shape[0], image_rgb.shape[1], 1), dtype=image_rgb.dtype)
            image_rgba = cv2.merge((image_rgb, alpha_channel))
        else:
            image_rgba = image_rgb

        
        # Convert RGBA to BGRA
        image_bgra = cv2.cvtColor(image_rgba, cv2.COLOR_RGBA2BGRA)
        dir_save = os.path.join(dir0, folder+"_cmmd")
        os.makedirs(dir_save, exist_ok=True)
        save_images(dir_save, image_bgra, dataname)
        
    return dir_save
        
def compute_cmmd(
    ref_dir, 
    eval_dir, 
    batch_size = 4, 
    max_count = -1
):
  """Calculates the CMMD distance between reference and eval image sets.

  Args:
    ref_dir: Path to the directory containing reference images.
    eval_dir: Path to the directory containing images to be evaluated.
    batch_size: Batch size used in the CLIP embedding calculation.
    max_count: Maximum number of images to use from each directory. A
      non-positive value reads all images available except for the images
      dropped due to batching.

  Returns:
    The CMMD value between the image sets.
  """
  embedding_model = embedding.ClipEmbeddingModel()
  ref_embs = io_util.compute_embeddings_for_dir(
      ref_dir, embedding_model, batch_size, max_count
  )
  eval_embs = io_util.compute_embeddings_for_dir(
      eval_dir, embedding_model, batch_size, max_count
  )
  val = distance.mmd(ref_embs, eval_embs)
  return np.asarray(val)

def main(argv):
  if len(argv) != 5:
    raise app.UsageError('Too few/too many command-line arguments.')
  _, dir1, dir2, eta, zeta = argv
  
  # Create logger
  script_dir = os.path.dirname(__file__)
  logger_name = f'"cmmd_results_zeta_{zeta}_eta_{eta}.log'
  log_dir = os.path.join(script_dir, "log")
  
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  
  
  # Configure the rotating file handler
  log_file = os.path.join(log_dir, logger_name)
  if os.path.isdir(log_file):
    raise IsADirectoryError(f"The specified log file path is a directory: {log_file}")

  
  dir1_new = rewrite_images(dir1)
  dir2_new = rewrite_images(dir2)
  cmmd_val = compute_cmmd(dir1_new, dir2_new, _BATCH_SIZE.value, _MAX_COUNT.value)
  shutil.rmtree(dir1_new)
  shutil.rmtree(dir2_new)
  save_dir = dir1.rsplit("/",1)[0]
  save_dir = os.path.join(save_dir, f"cmmd_results_zeta_{zeta}_eta_{eta}.csv")
  print(save_dir)
  # save data in a csv file 
  print(f"cmmd_results zeta {zeta} eta {eta}: {cmmd_val}")
  fid_pd = pd.DataFrame({"cmmd":[cmmd_val]})
  fid_pd.to_csv(save_dir, mode='a', header=not os.path.exists(save_dir))
  

if __name__ == '__main__':
  app.run(main)
