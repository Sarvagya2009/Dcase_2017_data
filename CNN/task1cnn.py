# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 16:29:15 2021

@author: Sarvagya Kumar
Core file
"""

import data_shifter #for making train test validation directories
import features #for making and saving log mel band energy features and saving them as npy files
import callrows #for yielding standardized batches of X and y to the model
import evaluate #for creating files with evaluation and development data accuracies
from tqdm import *
import sklearn as sk
import tensorflow as tf
import pandas as pd
import keras
import numpy as np
import os
#tf.keras.backend.clear_session()
tf.keras.backend.set_image_data_format('channels_last') #set format as per input data requirement
print(tf.keras.backend.image_data_format())



folds= [1,3] #adjust the number of folds as per user requirments



"""Till here we have copies the data into appropriate directory"""

DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/dir'
#create object for generate features class in features.py


#for each fold
for fold in folds:
    
    should_copy= False
    obj=data_shifter.meta2data(fold,should_copy)#obj is an object of datashifter class
    obj.copy() #calls function to make directory

    feature=features.generate_features(fold) #create an object for class features 
    
    #from train, test, validation metadata, extract rows and call make signal function in features.py
    for row in tqdm(obj.train.itertuples(), total=len(obj.train)):
        feature.make_signal(f'{DATA_PATH}/fold{fold}/train/{row.scene}/{row.file}', f'{DATA_PATH}/fold{fold}/train/{row.file}.logmelband.npy')
    for row in tqdm(obj.test.itertuples(), total=len(obj.test)):
        feature.make_signal(f'{DATA_PATH}/fold{fold}/test/{row.file}', f'{DATA_PATH}/fold{fold}/test/{row.file}.logmelband.npy')
    
    """ Till here we have made log mel band energies for all audios """
    
    #create object to get training feature matrix from create_dataset class of call rows
    train_obj= callrows.create_dataset(fold, obj.train, 'train')
    
    #next 2 lines are only for checking the dimensions of input data to CNN
    #_,y=next(train_obj.iterbatches(len(obj.train), obj.train, obj.label_encoder))
    #print(y.shape) #for training it is (3510, 1, 40, 501) where bands is 40
    
    #input shape: [batch_sz; band; frame_wnd; channel]
    input_shape= tf.keras.Input(shape=(40,501,1))
    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Conv2D(100, kernel_size=(2,2), activation='relu', input_shape=[40,501,1], kernel_regularizer=tf.keras.regularizers.L2(0.02), name='conv'))
    cnn.add(tf.keras.layers.BatchNormalization(axis=1))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add(tf.keras.layers.Conv2D(100, kernel_size=(2,2), activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.02)))
    cnn.add(tf.keras.layers.BatchNormalization(axis=1))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=2))
    cnn.add(tf.keras.layers.Dropout(0.2))
    
    #cnn.summary()  (None, 100, 9, 124) 
    cnn.add(tf.keras.layers.Flatten())
    cnn.add(tf.keras.layers.Dense(units=500, activation='relu'))
    cnn.add(tf.keras.layers.Dropout(0.2))
    cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
    cnn.add(tf.keras.layers.Dense(units=15, activation='softmax'))
    cnn.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['categorical_accuracy'])
    cnn.summary()
    batch_size=28
    
    """
    CAN BE CALLED AS CALLBACK TO VIEW METRICS ON TENSORBOARD
    tensorboard=tf.keras.callbacks.TensorBoard(
    log_dir='logs', histogram_freq=0, write_graph=True,
    write_images=True, update_freq='epoch', profile_batch=0,
    embeddings_freq=2, embeddings_metadata='metadata.tsv')
    """
    
    cnn.fit(x=train_obj.iterbatches(batch_size, obj.train, obj.label_encoder), epochs=40, verbose=1, steps_per_epoch=(len(obj.train)//batch_size), 
            max_queue_size=10)
    cnn.save(f'models/fold{fold}_model') #save the model
    
    
    
    """save development data predictions in a format acceptable for sed eval"""
    #in the following code we call training features of the fold and use our model to make predictions
    train_accuracy= callrows.create_dataset(fold, obj.train, 'train') 
    X, files, predictions = [], [], []
    for row in tqdm(obj.train.itertuples(), total=len(obj.train)):
        spec= train_accuracy._load_spec(f'{DATA_PATH}/fold{fold}/train/{row.file}.logmelband.npy')
        X.append(spec)
        _,filename=os.path.split(f'{DATA_PATH}/fold{fold}/train/{row.scene}/{row.file}')
        files.append(f'audio/{filename}')
    X = np.stack(X)
    mean= np.mean(X)
    std= np.std(X)
    X -= mean
    X /= std
    predictions = cnn.predict(X)
    predictions = np.argmax(predictions, axis=1)
    predictions=list(obj.label_encoder.inverse_transform(predictions)) #reverse encode the classes
    
    #write the predictions into a file in the format acceptable to sed_eval
    keys= ['scene_label', 'file']
    data=[predictions, files]
    #make a list of dictionary in the format: [{'scene_label': 'beach', 'file': 'audio/b020_90_100.wav'}]
    estimated=[]
    for i in range(len(predictions)):
        estimated.append(dict(zip(keys,[data[0][i], data[1][i]])))
    file= open(f'fold{fold}_estimated_development.txt', 'wt') #write the list of dictionary into this file 
    file.write(str(estimated))
    file.close()
    #call the accuracy function from evaluate file to compare labels and generate development accuracy
    dev= evaluate.accuracies(fold, 'train')
    dev.evaluate(estimated)
    
    """___________________________________________________________________________________________"""
    
    
    
    """save evaluation data predictions in a format acceptabel for sed eval""" 
    #we repeat the exact same process for the evaluation data
    evaluation_accuracy= callrows.create_dataset(fold, obj.test, 'test')
    X, files, predictions = [], [], []
    for row in tqdm(obj.test.itertuples(), total=len(obj.test)):
        spec= train_accuracy._load_spec(f'{DATA_PATH}/fold{fold}/test/{row.file}.logmelband.npy')
        X.append(spec)
        _,filename=os.path.split(f'{DATA_PATH}/fold{fold}/test/{row.file}')
        files.append(f'audio/{filename}')
    X = np.stack(X)
    mean= np.mean(X)
    std= np.std(X)
    X -= mean
    X /= std
    predictions = cnn.predict(X)
    predictions = np.argmax(predictions, axis=1)
    predictions=list(obj.label_encoder.inverse_transform(predictions))
    keys= ['scene_label', 'file']
    data=[predictions, files]
    estimated=[]
    for i in range(len(predictions)):
        estimated.append(dict(zip(keys,[data[0][i], data[1][i]])))
    file= open(f'fold{fold}_estimated_evaluation.txt', 'wt')
    file.write(str(estimated))
    file.close()
    dev= evaluate.accuracies(fold, 'evaluate')
    dev.evaluate(estimated)
    
    #reset the metrics of the model while we go to the next fold
    cnn.reset_metrics()