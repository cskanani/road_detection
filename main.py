from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from keras.utils import plot_model

import skimage.io as io
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pickle

def unet(input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2DTranspose(512, (2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop5)
    merge6 = concatenate([drop4,up6],axis=3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(256,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    merge7 =concatenate([conv3,up7],axis=3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(128,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    merge8 =concatenate([conv2,up8],axis=3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2DTranspose(64,(2,2),strides=(2,2), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    merge9 =concatenate([conv1,up9],axis=3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    model.summary()

    return model
 
def train_g(img_dir, msk_dir,gm = True):
    img_a = []
    msk_a = []
    for fn in os.listdir(img_dir):
        img = io.imread(img_dir+fn,as_gray = gm)
        img = np.reshape(img,img.shape + (1,)) if gm else img
        img_a.append(img)
    img_a = np.array(img_a)
    for fn in os.listdir(msk_dir):
        msk = io.imread(msk_dir+fn,as_gray = gm)
        msk = np.reshape(msk,msk.shape + (1,)) if gm else msk
        msk_a.append(msk)
    msk_a = np.array(msk_a)
    return img_a,msk_a

def train_generator(batch_size,train_dir,img_dir,msk_dir,cm ="grayscale",target_size=(512, 512),seed = 1):
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    image_generator = image_datagen.flow_from_directory(train_dir,classes=[img_dir],class_mode=None,color_mode = cm,batch_size = batch_size,target_size=target_size,seed = seed)
    mask_generator = mask_datagen.flow_from_directory(train_dir,classes=[msk_dir],class_mode = None,color_mode = cm,batch_size = batch_size,target_size=target_size,seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        yield (img,mask)

def test_generator(num_image,test_dir,cm='grayscale'):
    for i in range(1,num_image+1):
        img = io.imread(os.path.join(test_dir,"%d.jpg"%i),as_gray = cm)
        img = img / 255
        img = np.reshape(img,(1,)+img.shape+(1,))
        yield img
    

train_data = train_generator(2,'../data/train','images','labels')
test_data = test_generator(49,'../data/test/images')

model = unet()
#model_checkpoint = ModelCheckpoint('road_det_20.hdf5', monitor='loss', save_best_only=True)
#model.fit_generator(train_data,steps_per_epoch=1000,epochs=20,callbacks=[model_checkpoint])
model.load_weights('road_det_20.hdf5')
    
results = model.predict_generator(test_data,49,verbose=1)
pickle.dump(results,open('20_results.np','wb'))
