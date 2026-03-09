'''
MS - Artificial Intelligence and Machine Learning
Course: CSC510: Foundations of Artificial Intelligence
Module 8: Portfolio Project
Professor: Dr. Bingdong Li
Created by Mukul Mondal
January - February 2026
'''

# CNN deep learning implementation to detect Brain Tumor in image.

import os
from os import system, name
from pathlib import Path
import tensorflow as tf
from tensorflow.python.keras import Sequential
keras = tf.keras   #from tensorflow import keras
import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # pip install scikit-learn
from sklearn.preprocessing import OneHotEncoder
from PIL import Image   # pip install Pillow


# Part (b): Prediction based on Patient's MRI Scan file and the pattern within the image.
# Implementation class for CNN: model creation, training and prediction.
class MriBrainTumorCNN:
    def __init__(self):
        self.notumorImgDir: str = ""  # path for the non-tumor image files
        self.tumorImgDir: str = ""    # path for the tumor image files
        #self.imgSize = (128, 128)
        self.encoder = OneHotEncoder()
        self.encoder.fit([[0], [1]])  # 0 = tumor, 1: no tumor
        # create 3 lists
        self.fdata = []   # contains numpy equivalent of each image
        self.paths = []   # path of the corresponding image
        self.results = [] # test result of the corresponding image
        self.model: tf.keras.Model = None  # cnn deep learning model

    # sets to set paths for image files
    def SetImgDataPaths(self, notumorImgDir: str, tumorImgDir: str):
        self.notumorImgDir = notumorImgDir
        self.tumorImgDir = tumorImgDir
    
    # Prepares data for the CNN from raw image files
    def PreprocessAndLoadData(self):
        if self.notumorImgDir is None or len(self.notumorImgDir.strip()) < 1:
            print("Input data Error:", self.notumorImgDir)
            return
        if self.tumorImgDir is None or len(self.tumorImgDir.strip()) < 1:
            print("Input data Error:", self.tumorImgDir)
            return
        # process 'tumor' image files
        for r,d,f in os.walk(self.tumorImgDir):
            for file in f:
                if '.jpg' in file:
                    self.paths.append(os.path.join(r, file))
        
        for path in self.paths:
            img = Image.open(path)
            img = img.resize((128,128))
            img = np.array(img)
            if(img.shape ==(128,128,3)):
                self.fdata.append(np.array(img))
                self.results.append(self.encoder.transform([[0]]).toarray())  # 0 = tumor, 1: no tumor

        # process 'no tumor' image files
        self.paths = []
        for r,d,f in os.walk(self.notumorImgDir):
            for file in f:
                if '.jpg' in file:
                    self.paths.append(os.path.join(r, file))
        
        for path in self.paths:
            img = Image.open(path)
            img = img.resize((128,128))
            img = np.array(img)
            if(img.shape ==(128,128,3)):
                self.fdata.append(np.array(img))
                self.results.append(self.encoder.transform([[1]]).toarray())  # 0 = tumor, 1: no tumor

        self.fdata = np.array(self.fdata)
        self.results = np.array(self.results)
        self.results = self.results.reshape(self.fdata.shape[0], 2)
        #print(self.fdata.shape) # ok  (1221, 128, 128, 3)
        #print(self.fdata.shape[0]) # ok  1221       
        #print("len(paths) = ", len(self.paths)) # ok
        return
    
    # Initilizes the training and test data set; 
    #     epoch count for the model training and batch size for the training
    # It calls the model creation function and saves the model as this class's member.
    # does the model filt and calls the function show the models stat details.
    def InitModelAndTrain(self, epchs: int = 30, btchSize: int = 40):
        if epchs < 1 or btchSize < 2:
            epchs = 30     # set to default
            btchSize = 40  # set to default

        self.model = self.CreateModel()
        # Splitting the data into Training & Testing
        x_train, x_test, y_train, y_test = train_test_split(self.fdata, self.results, test_size=0.2, shuffle=True, random_state=0)

        history_model_fit = self.model.fit(x_train, y_train, epochs=epchs, batch_size=btchSize, verbose=1, validation_data =(x_test, y_test))
        self.ShowModelHistory(history_model_fit)
        return
    
    # Caller calls with the image file as argument for the prediction.
    # return the output as: ("Yes/No", "confidence percent".
    def predictImage(self, imgFile: str):
        if imgFile is None or len(imgFile.strip()) == 0:
            print("Invalid image file path.")
            return None, None
        self.showImage(imgFile.strip())
        img = Image.open(imgFile)
        x = np.array(img.resize((128,128)))
        x = x.reshape(1, 128, 128, 3)
        res = self.model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        YesNo: str = 'Unknown'
        if classification == 0:
            YesNo = "Yes" #'Tumor.'
        else:
            YesNo = "No"  #'Not Tumor.'
        
        #YesNo, PctConfidebce = predictImage(model=model3, imgFile=ffile)
        #if YesNo is not None and PctConfidebce is not None:
        #print(ffile, " ==> " , str(PctConfidebce) + '% Confidence this is: ' + YesNo)
        #print(imgFile, " ==> " ,"Tumor: " + YesNo + ", Confidence: " + str(res[0][classification]*100) + '%')
        return YesNo, str(res[0][classification]*100)

    # Creates 'model' of type 'Sequential' using 'keras' library.
    # We need here sequential layered model because from input layer to hidden layer to output layer
    #    information and execution procceds.
    # input_shape = (128,128,3) because that's how I did the image file preprocessing.
    # To keep it simple, I used the activation function: relu.
    # kernel_size=(2,2) used for better detection and prediction.
    # loss function: 'categorical_crossentropy' used because, meodel's output should be  categorical(tumor or NoTumor) 
    # for parameter details, please look in model summary table.
    #
    # Batch normalization is a technoque for training very deep neural networks that standardizes the inputs
    # to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically
    # reducing the number of training epochs required to train deep networks.
    #
    # returns the created model
    def CreateModel(self) -> Sequential:
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(2,2), input_shape=(128, 128, 3), padding="Same" ))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding="Same" ))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding="Same" ))
        model.add(keras.layers.Conv2D(filters=64, kernel_size=(2,2), activation='relu', padding="Same" ))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
        model.add(keras.layers.Dropout(0.25))

        model.add(keras.layers.Flatten())  # convert images into 1-D array
        
        model.add(keras.layers.Dense(512, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(2, activation='softmax'))

        model.compile( 
            optimizer='Adamax',             # optimizer=keras.optimizers.Adam(1e-4), 
            loss='categorical_crossentropy' #, x  'categorical_crossentropy', 'sparse_categorical_crossentropy'
                                            # metrics=['accuracy'] 
        )
        print(model.summary())
        #print("model creation : Success")
        return model

    # It shows loss and it's changes over epoch in UI.
    def ShowModelHistory(self, history_model_fit):
        dict_hist = history_model_fit.history
        plt.figure(figsize = (6,4))
        plt.plot(dict_hist['loss'])
        plt.plot(dict_hist['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Test', 'Validation'], loc='upper right')
        plt.show()  # plt.show(block=False)
        return    

    # It shows the image in UI for the file: imgFile.
    def showImage(self, imgFile: str):
        if imgFile is None or len(imgFile.strip()) == 0:
            print("Invalid image file path.")
            return
        img = Image.open(imgFile) # Image.open(r'./data/mri1/train/no_tumor/image (10).jpg')
        img_array = np.array(img)
        # print(img_array.shape)  # ok (128, 128, 3)
        plt.imshow(img_array)
        plt.axis("off")
        plt.show() # plt.show(block=False)
        return