import tensorflow as tf, pickle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard

from unet import get_model
from wsi_to_patches_1 import wsi_data_split, make_patches
from Data_Preprocessing_2 import train_gen, batch_plot

if __name__=="__main__":

    # Preprocessing WSIs and masks to mold them into a suitable shape
    training_split = 0.8
    shape = (512, 512)
    train_dir = './train/'
    val_dir = './val/'

    wsi_data_split(training_split)
    make_patches(train_dir, shape)
    make_patches(val_dir, shape)


    # Making training and validation batches to feed to model training
    train_img_dir = './train/patches/images/'
    train_mask_dir = './train/patches/masks/'
    val_img_dir = './val/patches/images/'
    val_mask_dir = './val/patches/masks/'
    batch_size = 32
    target_shape = shape

    train_batch = train_gen(train_img_dir, train_mask_dir, batch_size, target_shape, 1)
    val_batch = train_gen(val_img_dir, val_mask_dir, batch_size, target_shape)

    train_imgs, train_mask = next(train_batch)
    val_imgs, val_mask = next(val_batch)

    batch_plot(train_imgs)
    batch_plot(train_mask)
    batch_plot(val_imgs)
    batch_plot(val_mask)

    # Generating Model and Fitting it over the dataset and saving data into Train_History folder
    store_dir = './train_history/'

    if os.path.isdir(store_dir) is False:
        os.makedirs(store_dir)

    no_of_classes = 1
    model = get_model(shape, no_of_classes)
    model.summary()


    train_imgs = './train/patches/images/images/'
    val_imgs = './val/patches/images/images/'


    lr = 1e-3
    n_epochs = 30
    opt = tf.keras.optimizers.Adam(lr)
    train_steps = 1+len(os.listdir(train_imgs))/batch_size
    val_steps = 1+len(os.listdir(val_imgs))/batch_size

    metrics = ["acc", tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tf.keras.metrics.MeanIoU(2)]
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=metrics)
    callbacks = [
        ModelCheckpoint(store_dir+'model.h5', save_best_only=True),
        CSVLogger(store_dir+"training_Catalog.csv"),
        TensorBoard(store_dir+'logs'),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    ]

    history = model.fit(train_batch,
                        epochs = n_epochs,
                        steps_per_epoch = train_steps,
                        validation_data = val_batch,
                        callbacks = callbacks,
                        validation_steps = val_steps
                        )

    with open(store_dir+'history.pkl', 'wb') as file_pi:
        pickle.dump(history.history, file_pi)