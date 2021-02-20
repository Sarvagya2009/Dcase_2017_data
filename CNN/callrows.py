# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 12:56:06 2021

@author: Sarvagya Kumar

This class shuffles the feature matrix and yields data in batches
"""
import pandas as pd
from tqdm import *
import numpy as np
import sklearn as sk


import keras
keras.backend.set_image_data_format('channels_first')




DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development/dir'
class create_dataset:
    #arguments are fold, dataframe and directory {train,test,validation}
    def __init__(self, fold, dataset, direc):
        self.X=0
        self.y=0
        self.spec=0
        self.fold=fold
        self.dataset=dataset
        self.label_encoder = sk.preprocessing.LabelEncoder()
        self.n_scenes = None
        self.mean=0
        self.std=0
        self.direc=direc
    
    #returns the saved log mel band energy whose path is stored in spec
    def _load_spec(self, spec):
        return np.load(spec).astype('float32')
    
    #shuffles the dataset and yields row object 
    @staticmethod
    def _iterrows(dataset):
        while True:
            for row in dataset.iloc[np.random.permutation(len(dataset))].itertuples():
                yield row
    
    #to iterrate through batches
    def iterbatches(self, batch_size, dataset):
        itrain = self._iterrows(self.dataset)
        while True:
            self.X, self.y = [], [] # for features and target classes
            
            #itterate through batchsize
            for i in range(batch_size):
                row = next(itrain)
                #load spec whose name is given bu row.file
                spec = self._load_spec(f'{DATA_PATH}/fold{self.fold}/{self.direc}/{row.file}.logmelband.npy')
                #label encode values for classification
                scene_id = self.label_encoder.fit_transform([row.scene])[0]
                
                """
                print(spec.shape, "spec shape") GIVES (40,501) (long mel band feature shape)
                """
                #stack feature matrices in X
                self.X.append(np.stack([spec]))
                """
                print(np.array(self.X).shape, "self.X shape") GIVES [(sample,1),40,501]
                """
                #catergorical encoding of classes
                self.y.append(keras.utils.to_categorical(scene_id, self.n_scenes).ravel()) #size= (batchsize,1)
                
            X = np.stack(self.X) #((length of batch,1) 40, 501
            y = np.array(self.y) #length of batch, 1
            #standardization of values before passing it to generator
            self.mean= np.mean(X)
            self.std= np.std(X)
            X -= self.mean
            X /= self.std
            """
            print(X.shape, "X shape") GIVES (3510, 1, 40, 501) X shape
            print(y.shape, "y shape") GIVES (3510, 1) y shape
            """
            yield X, y
            