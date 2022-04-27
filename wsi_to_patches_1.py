from patchify import patchify
import os, tifffile as tif, numpy as np, shutil, random, glob



def make_patches(dir_path, patch_shape):
    
    """
    Arguments:
        dir_path : Path of the directory containing wsi images and masks to make patches.
        patch_shape : shape of the individual patch in 2D e.g. (512,512)
    """
    
    
    img_path = dir_path+'images/'
    mask_path = dir_path+'masks/'
    
    total_imgs = len(os.listdir(img_path))
    imgs_list = glob.glob(img_path+'*')
    masks_list = glob.glob(mask_path+'*')
        
    
    if os.path.isdir(dir_path+'patches/') is False:
        os.makedirs(dir_path+'patches/images/images/')
        os.makedirs(dir_path+'patches/masks/masks/')
    
    patch_img_path = dir_path+'patches/images/images/'
    patch_mask_path = dir_path+'patches/masks/masks/'
        
        
    for x in range(total_imgs):
        img = tif.imread(imgs_list[x])
        mask = tif.imread(masks_list[x])
        
        img_patches = patchify(img, patch_shape+(3,), step=patch_shape[0])
        mask_patches = patchify(mask, patch_shape, step=patch_shape[0])
        num = 0
        for i in range(mask_patches.shape[0]):
            for j in range(mask_patches.shape[1]):
                cur_patch = mask_patches[i][j]
                if np.max(cur_patch) > 0:
                    tif.imwrite(patch_mask_path+f'{x}_{num}.tif', cur_patch*255, photometric='minisblack')
                    tif.imwrite(patch_img_path+f'{x}_{num}.tif', img_patches[i][j][0], photometric='rgb')
                    num += 1     


def wsi_data_split(split_ratio):
    """

    Arguments:

        split_ratio : training and validation ratio to divide data e.g. 0.8 means 80% training data and remaining is validation data.

    """

    img_dir = './images/'
    total_data = len(os.listdir(img_dir))
    val_data = total_data*(1-split_ratio)
    val_img_paths = random.sample(glob.glob(img_dir+'*'), int(val_data) )


    if os.path.isdir('./val/') is False:
        os.makedirs('./val/images/')
        os.makedirs('./val/masks/')

    x = [shutil.move(img, './val/images/') for img in val_img_paths]
    
    mask_names = os.listdir('./val/images/')
    y = [shutil.move(f'./masks/{name}', './val/masks/') for name in mask_names]

    if os.path.isdir('./train/') is False:
        os.makedirs('./train')
    
    shutil.move('./images', './train/')
    shutil.move('./masks', './train/')

    del x,y

if __name__ == "__main__":
    wsi_data_split(0.8)
    make_patches('./train/', (512,512))
    make_patches('./val/', (512,512))