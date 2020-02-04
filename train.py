import unet

import os
import math
import numpy as np

from skimage.util import view_as_windows, pad

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator


def train_generator(
    batch_size,
    train_dir,
    image_dir,
    mask_dir,
    color_mode='grayscale',
    original_image_size=(1500, 1500),
    patch_size=(256, 256),
    seed=1,
):
    """
    Used for generating and passing images and lables to the fit_generator for training the model
    
    Attributes:
        batch_size (int): size of the batches of data
        train_dir (str): path to the directory containing images and masks
        image_dir (str): name of the folder containing images
        mask_dir (str): name of the folde conatining masks
        color_mode (str): 'rgb' or 'grayscale'
        original_image_size (tuple): original images will be resized to this shape
        patch_size (tuple): original image will be divided into patches of this size
        seed (int): optional random seed for shuffling and transformations, should be kept same for lable and image
        
    Return:
        (image, mask) : yields a tuple containing image(numpy array) and mask(numpy array)
    """

    data_gen_args_image = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='constant',
        cval=255,
        rescale=1 / 255.
    )
    data_gen_args_mask = data_gen_args_image.copy()
    data_gen_args_mask['cval'] = 0

    image_datagen = ImageDataGenerator(**data_gen_args_image)
    mask_datagen = ImageDataGenerator(**data_gen_args_mask)

    image_generator = image_datagen.flow_from_directory(
        train_dir,
        classes=[image_dir],
        class_mode=None,
        color_mode=color_mode,
        batch_size=batch_size,
        target_size=original_image_size,
        seed=seed
    )
    mask_generator = mask_datagen.flow_from_directory(
        train_dir,
        classes=[mask_dir],
        class_mode=None,
        color_mode=color_mode,
        batch_size=batch_size,
        target_size=original_image_size,
        seed=seed
    )

    train_generator = zip(image_generator, mask_generator)

    pad_width_x_axis = original_image_size[0] % patch_size[0]
    pad_width_y_axis = original_image_size[1] % patch_size[1]
    pad_width = ((0, pad_width_x_axis), (0, pad_width_y_axis), (0, 0))
    image_pad_values = ((1., 1.), (1., 1.), (1., 1.))
    mask_pad_values = ((0., 0.), (0., 0.), (0., 0.))
    if color_mode == 'grayscale':
        patch_size = patch_size + (1, )
    else:
        patch_size = patch_size + (3, )
    step_size = patch_size

    patches_per_row = math.ceil(original_image_size[0] / patch_size[0])
    patches_per_col = math.ceil(original_image_size[1] / patch_size[1])
    patches_per_image = patches_per_row * patches_per_col
    image_patches_batch = np.zeros(
        (batch_size * patches_per_image, ) + patch_size
    )
    mask_patches_batch = np.zeros(
        (batch_size * patches_per_image, ) + patch_size
    )

    for (image_batch, mask_batch) in train_generator:
        image_patches_batch_i = 0

        mask_batch[mask_batch > 0.5] = 1.
        mask_batch[mask_batch <= 0.5] = 0.

        for image_batch_i in range(image_batch.shape[0]):
            image = image_batch[image_batch_i]
            mask = mask_batch[image_batch_i]

            image = pad(
                image,
                pad_width=pad_width,
                mode='constant',
                constant_values=image_pad_values
            )

            mask = pad(
                mask,
                pad_width=pad_width,
                mode='constant',
                constant_values=mask_pad_values
            )

            image_patches = view_as_windows(
                image, window_shape=patch_size, step=step_size
            ).reshape((-1, ) + step_size)

            mask_patches = view_as_windows(
                mask, window_shape=patch_size, step=step_size
            ).reshape((-1, ) + step_size)

            for image_patches_i in range(image_patches.shape[0]):
                image_patches_batch[image_patches_batch_i] = image_patches[image_patches_i]
                mask_patches_batch[image_patches_batch_i] = mask_patches[image_patches_i]
                image_patches_batch_i += 1
        yield (image_patches_batch, mask_patches_batch)


# training the model
train_data = train_generator(10, 'data/train', 'images', 'labels')
validation_data = train_generator(10, 'data/val', 'images', 'labels')

model = unet.unet(input_size=(256, 256, 1))
model_checkpoint = ModelCheckpoint(
    'saved_model/unet_road_detection.hdf5',
    monitor='val_loss',
    save_best_only=True
)
model.fit_generator(
    train_data,
    steps_per_epoch=1000,
    validation_data=validation_data,
    validation_steps=200,
    epochs=10,
    callbacks=[model_checkpoint]
)
