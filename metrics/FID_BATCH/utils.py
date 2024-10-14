import matplotlib.image as mpimg
import cv2
import os

# save images
def save_images(dir, image, name):
    """
    Save an image to a specified directory with a given name.

    Args:
        dir (str): The directory where the image will be saved.
        image (numpy.ndarray): The image to be saved.
        name (str): The name of the image file.

    Returns:
        None
    """
    image_array = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(dir, name + '.png'), image_array)

# create batches
def create_batches(dir_images, num_patches):
    """
    Create batches of image patches from a directory of images.

    Args:
        dir_images (str): The directory path containing the images.
        num_patches (int): The number of patches to create from each axis.

    Returns:
        None
    """
    # Split the directory path to get the parent directory and create a new directory for the batches
    dir_images_batches = dir_images.rsplit('/', 1)
    dir_images_batches = os.path.join(dir_images_batches[0], dir_images_batches[1]+"_batch64")
 
    os.makedirs(dir_images_batches, exist_ok=True)
    
    # Get the list of files in the directory
    data = os.listdir(dir_images)

    # Read the first image to get the image size
    im_0 = mpimg.imread(os.path.join(dir_images, data[0]))
    im_size = im_0.shape[0]
    
    # Calculate the patch size based on the number of patches
    patch_size = int(im_size / num_patches)
    sized2 = int(patch_size / 2)
    
    # Iterate over each file in the directory
    ii = 0
    for filename in data:
        old_path_name = os.path.join(dir_images, filename)
        im = mpimg.imread(old_path_name) 
        
        # Check the number of channels in the image and remove the alpha channel if present
        im_channel = im.shape[-1]
        im = im[:,:,:3] if im_channel == 4 else im 
        
        idx = 0
        
        # Iterate over the image in patch_size steps to create patches
        for ii_x in range(0, im_size, patch_size):
            ii_x_next = ii_x + patch_size
            for ii_y in range(0, im_size, patch_size):
                ii_y_next = ii_y + patch_size
                patch = im[ii_x:ii_x_next, ii_y:ii_y_next]
                
                # Save the patch image
                save_images(dir_images_batches, patch, "{}_{:05}_{:06}".format(filename, ii, idx))
                ii += 1
                idx += 1
                
        # Iterate over the image in patch_size steps with an offset to create overlapping patches
        for cc_x in range(sized2, im_size - sized2, patch_size):
            cc_x_next = cc_x + patch_size
            for cc_y in range(sized2, im_size - sized2, patch_size):
                cc_y_next = cc_y + patch_size
    
                patch = im[cc_x:cc_x_next, cc_y:cc_y_next]
                
                # Save the patch image
                save_images(dir_images_batches, patch, "{}_{:05}_{:06}".format(filename, ii, idx))
                ii += 1
                idx += 1
    
    return dir_images_batches
                
                
            
        
    