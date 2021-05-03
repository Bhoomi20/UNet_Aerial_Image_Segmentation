import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import cv2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
def addaptive_histogram(img,clahe):
    H_list=[];
    if len(img.shape)==3:
        r,g,b = cv2.split(img)
        lit = [r,g,b]
        for img1 in lit:          
            equ = clahe.apply(img1)
            H_list.append(equ)
        H_img = cv2.merge((H_list[0],H_list[1],H_list[2]))
    else:
        H_img = clahe.apply(img)
        
    return H_img

# Set some parameters
IMG_WIDTH = 64
IMG_HEIGHT = 64
IMG_CHANNELS = 3

img_PATH = './Dataset5/Original Image/'
seg_PATH = './Dataset5/Targets Image/'#enter path to testing data

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

print("Imported all the dependencies")
# Get and resize train images and masks
X_train = np.zeros((len(os.listdir(img_PATH)), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(os.listdir(seg_PATH)), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)

print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
n=0
for img_dir1 in os.listdir(img_PATH):
    #print(img_dir1)
    img = imread(img_PATH + img_dir1)[:,:,:IMG_CHANNELS]
    img = addaptive_histogram(img,clahe)
    
    img90 = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    img90 = np.uint8(resize(img90, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True))
    
    img180 = cv2.rotate(img, cv2.ROTATE_180)
    img180 = np.uint8(resize(img180, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True))
    
    img270 = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img270 = np.uint8(resize(img270, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True))
    
    img = np.uint8(resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True))
    
    image_list=[img,img90,img180,img270];

    for add_img in image_list:
        X_train[n] = add_img
        n+=1
    #if n==100:
    #    break
        
n=0
for img_dir1 in os.listdir(seg_PATH):
    #print(img_dir1)
    mask = imread(seg_PATH + img_dir1)
    mask[mask>0]=1
    mask = np.expand_dims(resize(mask, (IMG_HEIGHT, IMG_WIDTH), mode='constant',preserve_range=True), axis=-1)
    mask = np.uint8(mask)
    Y_train[n] = mask
    n+=1
    #if n==100:
    #    break
"""
for i in range(0,9):
    img11= imread(img_PATH + os.listdir(img_PATH)[0])[:,:,:IMG_CHANNELS]
    img11 = cv2.resize(img11,(256,256))

    img22= mask = imread(seg_PATH + os.listdir(seg_PATH)[0])
    img22[img22>=1]=255
    img22 = cv2.resize(img22,(256,256))
    img22 = cv2.merge((img22,img22,img22))
    Hori = np.concatenate((img11, img22), axis=1)
    cv2.imshow('HORIZONTAL', Hori)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""    
# Get and resize test images
X_test = X_train

print(' Data Processing Done!')


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def U_NET():
    # Build U-Net model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = Lambda(lambda x: x / 255) (inputs)

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
    c5 = Dropout(0.3) (c5)
    c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.2) (c6)
    c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.2) (c7)
    c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
    c8 = Dropout(0.1) (c8)
    c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
    c9 = Dropout(0.1) (c9)
    c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    return model


model = U_NET()

# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0., batch_size=16, epochs=20, 
                    callbacks=[earlystopper, checkpointer])

