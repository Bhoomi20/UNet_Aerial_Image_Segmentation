import os
import time
import warnings
import numpy as np
import matplotlib
matplotlib.use('agg')

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras import backend as K

import tensorflow as tf
import cv2
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
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

class automaticmaplabelling():
    def __init__(self,modelPath,full_chq,imagePath,width,height,channels):
        print(width)
        print(height)
        print(channels)
        self.modelPath=modelPath
        self.full_chq=full_chq
        self.imagePath=imagePath
        self.IMG_WIDTH=width
        self.IMG_HEIGHT=height
        self.IMG_CHANNELS=channels
        self.model = self.U_net()
    
    
    
 # Insertion over union ( for merging overlapped segmented )
    def mean_iou(self,y_true, y_pred):
        prec = []
        for t in np.arange(0.5, 1.0, 0.05):
            y_pred_ = tf.to_int32(y_pred > t)
            score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
            K.get_session().run(tf.local_variables_initializer())
            with tf.control_dependencies([up_opt]):
                score = tf.identity(score)
            prec.append(score)
        return K.mean(K.stack(prec), axis=0)

    def U_net(self):
        # Build U-Net model
        inputs = Input((self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS))
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
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[self.mean_iou])
        model.load_weights(self.modelPath)
        #model.summary()
        return model

    def prediction(self):
        img=cv2.imread(self.imagePath,0)
        maskl = cv2.threshold(img,190,255,cv2.THRESH_BINARY)
        img=np.expand_dims(img,axis=-1)
        x_test= np.zeros((1, self.IMG_HEIGHT, self.IMG_WIDTH, self.IMG_CHANNELS), dtype=np.uint8)
        #testimg=resize(img,(self.IMG_HEIGHT,self.IMG_WIDTH),mode='constant',preserve_range=True)
        x_test[0]=img
        preds_test= self.model.predict(x_test, verbose=1)
        #print(preds_test)
        preds_test = ((preds_test >= 0.5))
        mask1=np.uint8(preds_test[0])
        mask1[mask1==1]=255
        #cv2.imshow('',maskl[1])
        #cv2.waitKey(0)
        return x_test[0],maskl[1]

def startProcessing(pre_disaster_image_path="./pre.png",post_disaster_image_path="./post.png"):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # Select Image
    filename = pre_disaster_image_path
    # Read Image From Directory
    img1 = cv2.imread(filename)
    H_img1 = addaptive_histogram(img1,clahe)
    test_image_name1 = "pre_disaster.png"
    H_img1 = cv2.resize(H_img1,(512,512))
    cv2.imwrite(test_image_name1,H_img1)

    # Select Image
    filename = post_disaster_image_path
    # Read Image From Directory
    img2 = cv2.imread(filename)
    H_img2 = addaptive_histogram(img2,clahe)
    test_image_name2 = "post_disaster.png"
    H_img2 = cv2.resize(H_img2,(512,512))
    cv2.imwrite(test_image_name2,H_img2)
    
    automaticmaplabellingobj= automaticmaplabelling('model-dsbowl2018-1.h5',True,test_image_name1,512,512,3)
    testimg,mask1 = automaticmaplabellingobj.prediction()

    automaticmaplabellingobj= automaticmaplabelling('model-dsbowl2018-1.h5',True,test_image_name2,512,512,3)
    testimg,mask2 = automaticmaplabellingobj.prediction()

    Input_image = np.concatenate((cv2.resize(img1,(512,512)), cv2.resize(img2,(512,512))), axis=1)
    Enhance_Image = np.concatenate((H_img1, H_img2), axis=1)
    segmented_Image = np.concatenate((cv2.resize(mask1,(512,512)), cv2.resize(mask2,(512,512))), axis=1)
    mask1=np.array(mask1)
    mask2 = np.array(mask2)
    
    bw_image = np.uint8(((mask1-mask2)))
    
    mark_damage_region = cv2.merge((bw_image,bw_image,bw_image))
    
    t = int(time.time())
    image_dir = os.path.join(".\\static\\results\\", str(t))
    input_image_path = os.path.join(image_dir, "input_image_path.png")
    Enhance_Image_path = os.path.join(image_dir, "Enhance_Image_path.png")
    segmented_Image_path = os.path.join(image_dir, "segmented_Image_path.png")
    mark_damage_region_path = os.path.join(image_dir, "mark_damage_region_path.png")
    os.mkdir(image_dir)
    #Visualization
    cv2.imwrite(input_image_path,Input_image)
    cv2.imwrite(Enhance_Image_path,Enhance_Image)
    cv2.imwrite(segmented_Image_path,segmented_Image)
    cv2.imwrite(mark_damage_region_path,mark_damage_region)

    # cv2.imshow('Input Image',Input_image)
    # cv2.waitKey(0)
    # cv2.imshow('Pre_processed Image',Enhance_Image)
    # cv2.waitKey(0)
    # cv2.imshow('Segmented Image',segmented_Image)
    # cv2.waitKey(0)
    # cv2.imshow('Dmaged Region',mark_damage_region)
    # cv2.waitKey(0)
    #cv2.destroyAllWindows()

    bw_image[bw_image>=1]=1
    bw_image=bw_image.flatten();
    Total_change= np.sum(bw_image)

    mask1=mask1.flatten()
    mask1[mask1>=1]=1
    Pre_unchangedpixel=np.sum(mask1)
    
    percentage_damage = (Total_change/(Pre_unchangedpixel))*100
    print("Total Percent of Damage in segmented Region ", percentage_damage,' %')
    # print("Total Percent of Damage in segmented Region ", percentage_damage,' %')
    # percentage_damage = (Total_change/(512*512))*100
    # print("Total Percent of Damage in completed Image ", percentage_damage,' %')
    
    return percentage_damage,input_image_path,Enhance_Image_path,segmented_Image_path,mark_damage_region_path
    # cv2.imwrite("resized_mask.png",bw_image)

if __name__ == "__main__":
    startProcessing()