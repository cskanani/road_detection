from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as keras
#from keras.utils import plot_model

import skimage.io as io
import numpy as np
import pickle

import cv2
import numpy as np


def unet(input_shape = (512,512,1)):
    """
    This function impliments u-net architecture using keras functional API
    
    Attributes:
        input_shape (tuple): size of the input
        
    Return:
        model : an object of class Model
    """
    inputs = Input(input_shape)
    
    conv1 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(inputs)
    conv1 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(conv1)
    
    conv2 = Conv2D(128, 3, padding='same', strides=(2,2), activation='relu')(conv1)
    conv2 = Conv2D(128, 3, padding='same', strides=(1,1), activation='relu')(conv2)
    
    conv3 = Conv2D(256, 3, padding='same', strides=(2,2), activation='relu')(conv2)
    conv3 = Conv2D(256, 3, padding='same', strides=(1,1), activation='relu')(conv3)
    
    conv4 = Conv2D(512, 3, padding='same', strides=(2,2), activation='relu')(conv3)
    conv4 = Conv2D(512, 3, padding='same', strides=(1,1), activation='relu')(conv4)
    
    conv5 = Conv2D(1024, 3, padding='same', strides=(2,2), activation='relu')(conv4)
    conv5 = Conv2D(1024, 3, padding='same', strides=(1,1), activation='relu')(conv5)
    
    up6 = UpSampling2D(size=(2, 2))(conv5)
    merge6 = concatenate([up6, conv4], axis=3)
    conv6 = Conv2D(512, 3, padding='same', strides=(1,1), activation='relu')(merge6)
    conv6 = Conv2D(512, 3, padding='same', strides=(1,1), activation='relu')(conv6)
    
    up7 = UpSampling2D(size=(2, 2))(conv6)
    merge7 = concatenate([up7, conv3], axis=3)
    conv7 = Conv2D(256, 3, padding='same', strides=(1,1), activation='relu')(merge7)
    conv7 = Conv2D(256, 3, padding='same', strides=(1,1), activation='relu')(conv7)
    
    up8 = UpSampling2D(size=(2, 2))(conv7)
    merge8 = concatenate([up8, conv2], axis=3)
    conv8 = Conv2D(128, 3, padding='same', strides=(1,1), activation='relu')(merge8)
    conv8 = Conv2D(128, 3, padding='same', strides=(1,1), activation='relu')(conv8)
    
    up9 = UpSampling2D(size=(2, 2))(conv8)
    merge9 = concatenate([up9, conv1], axis=3)
    conv9 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(merge9)
    conv9 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(conv9)
    
    out = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input = inputs, output = out)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model
 

def unet_res(input_shape = (512,512,1)):
    """
    This function impliments u-net architecture with residual blocks using keras functional API
    
    Attributes:
        input_shape (tuple): size of the input
        
    Return:
        model : an object of class Model
    """
    
    inputs = Input(input_shape)
    
    conv1 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(inputs)
    conv1 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(conv1)
    shrt1 = Conv2D(64, 1, strides=(1,1))(inputs)
    res1 = add([shrt1, conv1])
    
    conv2 = Conv2D(128, 3, padding='same', strides=(2,2), activation='relu')(res1)
    conv2 = Conv2D(128, 3, padding='same', strides=(1,1), activation='relu')(conv2)
    shrt2 = Conv2D(128, 1, strides=(2,2))(res1)
    res2 = add([shrt2, conv2])
    
    conv3 = Conv2D(256, 3, padding='same', strides=(2,2), activation='relu')(res2)
    conv3 = Conv2D(256, 3, padding='same', strides=(1,1), activation='relu')(conv3)
    shrt3 = Conv2D(256, 1, strides=(2,2))(res2)
    res3 = add([shrt3, conv3])
    
    conv4 = Conv2D(512, 3, padding='same', strides=(2,2), activation='relu')(res3)
    conv4 = Conv2D(512, 3, padding='same', strides=(1,1), activation='relu')(conv4)
    shrt4 = Conv2D(512, 1, strides=(2,2))(res3)
    res4 = add([shrt4, conv4])
    
    conv5 = Conv2D(1024, 3, padding='same', strides=(2,2), activation='relu')(res4)
    conv5 = Conv2D(1024, 3, padding='same', strides=(1,1), activation='relu')(conv5)
    shrt5 = Conv2D(1024, 1, strides=(2,2))(res4)
    res5 = add([shrt5, conv5])
    
    up6 = UpSampling2D(size=(2, 2))(res5)
    merge6 = concatenate([up6, res4], axis=3)
    conv6 = Conv2D(512, 3, padding='same', strides=(1,1), activation='relu')(merge6)
    conv6 = Conv2D(512, 3, padding='same', strides=(1,1), activation='relu')(conv6)
    shrt6 = Conv2D(512, 1, strides=(1,1))(merge6)
    res6 = add([shrt6, conv6])
    
    up7 = UpSampling2D(size=(2, 2))(res6)
    merge7 = concatenate([up7, res3], axis=3)
    conv7 = Conv2D(256, 3, padding='same', strides=(1,1), activation='relu')(merge7)
    conv7 = Conv2D(256, 3, padding='same', strides=(1,1), activation='relu')(conv7)
    shrt7 = Conv2D(256, 1, strides=(1,1))(merge7)
    res7 = add([shrt7, conv7])
    
    up8 = UpSampling2D(size=(2, 2))(res7)
    merge8 = concatenate([up8, res2], axis=3)
    conv8 = Conv2D(128, 3, padding='same', strides=(1,1), activation='relu')(merge8)
    conv8 = Conv2D(128, 3, padding='same', strides=(1,1), activation='relu')(conv8)
    shrt8 = Conv2D(128, 1, strides=(1,1))(merge8)
    res8 = add([shrt8, conv8])
    
    up9 = UpSampling2D(size=(2, 2))(res8)
    merge9 = concatenate([up9, res1], axis=3)
    conv9 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(merge9)
    conv9 = Conv2D(64, 3, padding='same', strides=(1,1), activation='relu')(conv9)
    shrt9 = Conv2D(64, 1, strides=(1,1))(merge9)
    res9 = add([shrt9, conv9])
    
    out = Conv2D(1, 1, activation='sigmoid')(res9)

    model = Model(input = inputs, output = out)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model
 
    
def train_generator(batch_size,train_dir,img_dir,msk_dir,target_size=(512, 512),md='3d'seed = 1):
    """
    Used for generating and passing images and lables to the fit_generator for training the model
    
    Attributes:
        batch_size (int): size of the batches of data
        train_dir (str): path to the directory containing images and labels
        img_dir (str): name of the folder containing images
        msk_dir (str): name of the folde conatining masks/labels
        target_size (tuple):taget size of the images
        seed (int): optional random seed for shuffling and transformations, should be kept same for lable and image
        md (str): 1d or 3d, if used 3d two additional filters are appended to gray scale image, default is 3d
        
    Return:
        (img,mask) : yields a tuple containing image(numpy array) and label(numpy array)
    """
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    if(md = '1d'):
        image_generator = image_datagen.flow_from_directory(train_dir,classes=[img_dir],class_mode=None,
                                                        color_mode='grayscale',batch_size=batch_size,target_size=target_size,seed=seed)
        mask_generator = mask_datagen.flow_from_directory(train_dir,classes=[msk_dir],class_mode=None,
                                                        color_mode='grayscale',batch_size=batch_size,target_size=target_size,seed=seed)
        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            yield (img,mask)
    else:
        image_generator = image_datagen.flow_from_directory(train_dir,classes=[img_dir],class_mode=None,
                                                            batch_size=batch_size,target_size=target_size,seed=seed)
        mask_generator = mask_datagen.flow_from_directory(train_dir,classes=[msk_dir],class_mode=None,
                                                        color_mode='grayscale',batch_size=batch_size,target_size=target_size,seed=seed)
        train_generator = zip(image_generator, mask_generator)
        for (img,mask) in train_generator:
            img = cv2.cvtColor(img[0], cv2.COLOR_BGR2GRAY).astype('uint8')
            imgn = cv2.bitwise_not(img)
            imgw = cv2.adaptiveThreshold(imgn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
            img = np.dstack((img, imgn,imgw))
            img = img.reshape((1,)+img.shape)
            img = img / 255
            mask = mask / 255
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            yield (img,mask)
            
            
def test_generator(num_image,test_dir,md='3d'):
    """
    Used for generating and passing test images and lables to the predict_generator
    
    Attributes:
        num_image (int): number of images in test folder, NOTE : images in test folder must be numbered from 1 to the num_image, both included
        test_dir (str): path to the directory containing test images
        md (str): 1d or 3d, if used 3d two additional filters are appended to gray scale image, default is 3d
        
    Return:
        img : yields image(numpy array)
    """
    if(md = '1d'):
        for i in range(1,num_image+1):
            img = cv2.imread(os.path.join(test_dir,"%d.jpg"%i),0)
            img = img / 255
            img = np.reshape(img,(1,)+img.shape+(1,))
            yield img
    else:
        for i in range(1,num_image+1):
            img = cv2.imread(os.path.join(test_dir,"%d.jpg"%i),0)
            imgn = cv2.bitwise_not(img)
            imgw = cv2.adaptiveThreshold(imgn, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
            img = np.dstack((img, imgn,imgw))
            img = img.reshape((1,)+img.shape)
            img = img / 255
            yield img
    

#train_data = train_generator(1,'../data/train','images','labels')
test_data = test_generator(49,'../data/test/images')


#model = unet(input_shape = (512,512,3))
##model_checkpoint = ModelCheckpoint('unet_20_3d.hdf5', monitor='loss', save_best_only=True)
##model.fit_generator(train_data,steps_per_epoch=1000,epochs=20,verbose=2,callbacks=[model_checkpoint])

#model.load_weights('unet_20_3d.hdf5')    
#results = model.predict_generator(test_data,49,verbose=1)
#pickle.dump(results,open('unet_20_3d_results.np','wb'))



model = unet_res(input_shape = (512,512,3))
#model_checkpoint = ModelCheckpoint('unet_res_20_3d.hdf5', monitor='loss', save_best_only=True)
#model.fit_generator(train_data,steps_per_epoch=1000,epochs=20,verbose=2,callbacks=[model_checkpoint])

model.load_weights('unet_res_20_3d.hdf5')    
results = model.predict_generator(test_data,49,verbose=1)
pickle.dump(results,open('unet_res_20_3d_results.np','wb'))




#model.load_weights('road_det_20.hdf5')    
#results = model.predict_generator(test_data,49,verbose=1)
#pickle.dump(results,open('20_results.np','wb'))
