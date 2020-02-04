import unet

import os
import math
import numpy as np
import cv2

from skimage import img_as_ubyte
from skimage.util import view_as_windows, pad
from sklearn.metrics import classification_report


def get_prediction_masks(patch_predictions, original_image_size, patch_size):
    """
    Generates full prediction masks from mask patches
    
    Attributes:
        patch_predictions (np array): array containing masks generated for small image patches
        original_image_size (tuple): tuple specifying width and height of original images/masks
        patch_size (tuple): tuple specifying width and height of smaller image patches
        
    Return:
        prediction_masks (numpy array): an array of size ((num_images,) + original_image_size)
                                        here num_images = total test images
    """
    patches_per_row = math.ceil(original_image_size[0] / patch_size[0])
    patches_per_col = math.ceil(original_image_size[1] / patch_size[1])
    patches_per_image = patches_per_row * patches_per_col
    num_images = patch_predictions.shape[0] // patches_per_image
    prediction_size = (
        original_image_size[0] + (original_image_size[0] % patch_size[0]),
        original_image_size[1] + (original_image_size[1] % patch_size[1])
    )
    prediction_masks = np.zeros((num_images, ) + original_image_size)

    for k in range(num_images):
        mask_prediction = patch_predictions[k * patches_per_image:
                                            (k + 1) * patches_per_image]
        mask_prediction = mask_prediction.reshape(
            (patches_per_image, ) + patch_size
        )
        prediction = np.zeros(prediction_size)
        for i in range(patches_per_image):
            prediction[int(i / patches_per_col) * patch_size[1]:(
                int(i / patches_per_col) + 1
            ) * patch_size[1], (i % patches_per_row) * patch_size[0]:(
                (i % patches_per_row) + 1
            ) * patch_size[0]] = mask_prediction[i]
        prediction = prediction[:original_image_size[0], :original_image_size[0]
                               ]
        prediction[prediction > 0.5] = 1.
        prediction[prediction <= 0.5] = 0.
        prediction_masks[k] = prediction

    return prediction_masks


def test_generator(
    test_dir,
    image_dir,
    num_image=49,
    color_mode='grayscale',
    original_image_size=(1500, 1500),
    patch_size=(256, 256),
):
    """
    Used for generating and passing test images and lables to the predict_generator
    
    Attributes:
        test_dir (str): path to the directory containing images and masks
        image_dir (str): name of the folder containing images
        num_image (int): number of images in test folder,
                        NOTE : images in test folder must be numbered from 1 to the num_image, both included
        color_mode (str): 'rgb' or 'grayscale'
        original_image_size (tuple): original images will be resized to this shape
        patch_size (tuple): original image will be divided into patches of this size
        
    Return:
        image : yields image(numpy array)
    """

    if color_mode == 'grayscale':
        patch_size = patch_size + (1, )
    else:
        patch_size = patch_size + (3, )

    step_size = patch_size
    pad_width_x_axis = original_image_size[0] % patch_size[0]
    pad_width_y_axis = original_image_size[1] % patch_size[1]
    pad_width = ((0, pad_width_x_axis), (0, pad_width_y_axis), (0, 0))
    image_pad_values = ((1., 1.), (1., 1.), (1., 1.))

    for i in range(1, num_image + 1):
        if color_mode == 'grayscale':
            image = cv2.imread(
                os.path.join(test_dir, image_dir, '%d.jpg' % i), 0
            )
            image = cv2.resize(image, original_image_size)
            image = image.reshape(original_image_size + (1, ))

        else:
            image = cv2.imread(os.path.join(test_dir, image_dir, '%d.jpg' % i))
            image = cv2.resize(image, original_image_size)
        image = pad(
            image,
            pad_width=pad_width,
            mode='constant',
            constant_values=image_pad_values
        )
        image_patches = view_as_windows(
            image, window_shape=patch_size, step=step_size
        ).reshape((-1, ) + step_size)
        image_patches = image_patches / 255.

        yield image_patches


# generating road map for testing images
test_num_images = 49
original_image_size = (1500, 1500)
patch_size = (256, 256)

test_data = test_generator(
    'data/test',
    'images',
    num_image=test_num_images,
    original_image_size=original_image_size,
    patch_size=patch_size
)
model = unet.unet(input_size=(256, 256, 1))
model.load_weights('saved_model/unet_road_detection.hdf5')
patch_predictions = model.predict_generator(
    test_data, steps=test_num_images, verbose=1
)
prediction_masks = get_prediction_masks(
    patch_predictions=patch_predictions,
    original_image_size=original_image_size,
    patch_size=patch_size
)

# loading original labels/masks for score calculation
original_masks = np.zeros((test_num_images, ) + original_image_size)
for i in range(1, test_num_images + 1):
    mask = cv2.imread(os.path.join('data/test', 'labels', '%d.jpg' % i), 0)
    mask = cv2.resize(mask, original_image_size)
    mask = mask / 255.
    mask[mask > 0.5] = 1.
    mask[mask <= 0.5] = 0.
    original_masks[i - 1] = mask

# calculating scores
prediction_masks = prediction_masks.reshape(-1)
original_masks = original_masks.reshape(-1)
print(classification_report(original_masks, prediction_masks))
