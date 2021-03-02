# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 03:03:57 2021

@author: Sarvagya Kumar


Goal: To make training and test folders directory like:
    1. Train
    ----cafe
    ----beaches
    .....
    
    
    
    
    2. Test data
    ----audiofold.wav

"""

import os
import pandas as pd
from tqdm import *
import shutil
import sklearn as sk
import numpy as np

#datapath where the metadata and the audio files have been downloaded by the baseline system 
#(and where the new directory of data will be made)
DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development'
class meta2data:
    
    def __init__(self,fold, should_copy):
        self.fold = fold
        self.train = None
        self.test = None
        self.should_copy=should_copy
        self.label_encoder = sk.preprocessing.LabelEncoder()
        self.counter=0
        
    def copy(self):
        #This function enables us to get 2 column names- file and scene from metadata, in a dataframe
        def read_fold(filename):
            return pd.read_csv(f'{DATA_PATH}/evaluation_setup/{filename}',
                               sep='\t', names=['file', 'scene'],
                               converters={'file': lambda s: s.replace('audio/', '')})
    
    
    
    #read each metadata file and make a dataframe out of this
        self.train= read_fold(f'fold{self.fold}_train.txt')
        #concatenating labelled data into one dataframe
        #train = pd.concat([train, validation], ignore_index=True)
        self.test= read_fold(f'fold{self.fold}_test.txt')
        if self.counter==0:
            a=np.array(self.train.scene)
            b=self.label_encoder.fit(a)
            self.counter+=1
                
        if self.should_copy == True: #if and only if user has specified that files should be copied
                #iterate through each row of the dataframe with tqdm which provides us with a progress bar
            for row in tqdm(self.train.itertuples(), total=len(self.train)):
                    # if the label folder doesnt exist, make that folder
                if not os.path.exists(f'{DATA_PATH}/dir/fold{self.fold}/train/{row.scene}'):
                    os.makedirs(f'{DATA_PATH}/dir/fold{self.fold}/train/{row.scene}')
                    #copy from the downloaded directory to the new directory
                shutil.copy2(f'{DATA_PATH}/audio/{row.file}', f'{DATA_PATH}/dir/fold{self.fold}/train/{row.scene}')
               
                #iterate through each row of the dataframe with tqdm which provides us with a progress bar
            for row in tqdm(self.test.itertuples(), total=len(self.test)):
                    # if the test folder doesnt exist, make that folder
                if not os.path.exists(f'{DATA_PATH}/dir/fold{self.fold}/test'):
                    os.makedirs(f'{DATA_PATH}/dir/fold{self.fold}/test')
                    #copy from downloaded directory to test folder
                shutil.copy2(f'{DATA_PATH}/audio/{row.file}', f'{DATA_PATH}/dir/fold{self.fold}/test')
                    
               