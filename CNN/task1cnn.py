# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:29:15 2021

@author: Sarvagya Kumar
Core file
"""

import data_shifter #for making train test validation directories
import features #for making and saving log mel band energy features and saving them as npy files
import callrows #for yielding standardized batches of X and y to the model
from tqdm import *

import tensorflow as tf





folds= [1] #adjust the number of folds as per user requirments
should_copy= False
obj=data_shifter.meta2data(folds,should_copy)#obj is an object of datashifter class
obj.copy() #calls function to make directory

"""Till here we have copies the data into appropriate directory"""

DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/dir'
#create object for generate features class in features.py
feature=features.generate_features(folds)

#for each fold
for fold in folds:
    #from train, test, validation metadata, extract rows and call make signal function in features.py
    for row in tqdm(obj.train.itertuples(), total=len(obj.train)):
        feature.make_signal(f'{DATA_PATH}/fold{fold}/train/{row.scene}/{row.file}', f'{DATA_PATH}/fold{fold}/train/{row.file}.logmelband.npy')
    for row in tqdm(obj.validation.itertuples(), total=len(obj.validation)):
        feature.make_signal(f'{DATA_PATH}/fold{fold}/evaluate/{row.scene}/{row.file}', f'{DATA_PATH}/fold{fold}/evaluate/{row.file}.logmelband.npy')
    for row in tqdm(obj.test.itertuples(), total=len(obj.test)):
        feature.make_signal(f'{DATA_PATH}/fold{fold}/test/{row.file}', f'{DATA_PATH}/fold{fold}/test/{row.file}.logmelband.npy')
    
    """ Till here we have made log mel band energies for all audios """
    
    #create object to get training feature matrix from create_dataset class of call rows
    train_obj= callrows.create_dataset(fold, obj.train, 'train')
    
    #next 2 lines are only for checking the dimensions of input data to CNN
    X,_=next(train_obj.iterbatches(len(obj.train), obj.train))
    print(X.shape) #for training it is (3510, 1, 40, 501) where bands is 40
    
    #create CNN architecture
    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Conv2D(100, kernel_size=(2,2), activation='relu', input_shape=[1,40,501], kernel_regularizer=tf.keras.regularizers.L2(0.02)))
    cnn.add(tf.keras.layers.BatchNormalization(axis=1))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    cnn.add(tf.keras.layers.Dropout(0.2))
    
    cnn.add(tf.keras.layers.Conv2D(100, kernel_size=(2,2), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.02)))
    cnn.add(tf.keras.layers.BatchNormalization(axis=1))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    cnn.add(tf.keras.layers.Dropout(0.2))
    
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=500, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=15, activation='sigmoid'))
    cnn.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    cnn.summary()
    batch_size=32
    cnn.fit(x=train_obj.iterbatches(batch_size, obj.train), epochs=500, verbose=1)
    
    

   