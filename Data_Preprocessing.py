import tensorflow as tf, random
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

def train_gen(img_dir, mask_dir, batch_size, target_shape, batch_type = 0):
    """
    Arguments:
        img_dir : Training patch images directory
        mask_dir : Training patch images masks directory
        batch_size : Batch_size in which training occurs
        target_shape : Final output size of the image and mask
        batch_type : For training batch put 1 for validation and testing default is 0, used to control image augmentation
    """
    
    seed = 909
    aug = dict(horizontal_filp=True, 
                vertical_flip=True, 
                fill_mode='nearest', 
                width_shift_range=0.2, 
                height_shift_range=0.2, 
                brightness_range=[0.4,1.5], 
                zoom_range=0.3)

    kwargs = aug if batch_type else {}
    wsi_datagen = ImageDataGenerator(kwargs, rescale=1/255)
    mask_datagen = ImageDataGenerator(kwargs, rescale=1/255)

    wsi_gen = wsi_datagen.flow_from_directory(img_dir, 
                                               batch_size=batch_size, 
                                               target_size=target_shape, 
                                               class_mode=None, 
                                               seed=seed)

    mask_gen = mask_datagen.flow_from_directory(mask_dir, 
                                                batch_size=batch_size, 
                                                target_size=target_shape, 
                                                class_mode=None, 
                                                seed=seed, 
                                                color_mode='grayscale')

    return zip(wsi_gen, mask_gen)

def batch_plot(Data_arr):
    """
    Arguments:
        Data_arr : array of images and masks from particular batch of imageDataGenerator processed patches.
    """
    fig, axes = plt.subplots(1,10,figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip(Data_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    train_batch = train_gen('./train/patches/images/', './train/patches/masks/', 32, (512,512), 1)
    imgs, masks = next(train_batch)
    batch_plot(imgs)
    batch_plot(masks)
    # val_batch = train_gen('./val/patches/images/','./val/patches/masks/',32,(512,512))
