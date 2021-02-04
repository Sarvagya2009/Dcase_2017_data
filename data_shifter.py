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

#datapath where the metadata and the audio files have been downloaded by the baseline system 
#(and where the new directory of data will be made)
DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development'

#This function enables us to get 2 column names- file and scene from metadata, in a dataframe
def read_fold(filename):
            return pd.read_csv(f'{DATA_PATH}/evaluation_setup/{filename}',
                               sep='\t', names=['file', 'scene'],
                               converters={'file': lambda s: s.replace('audio/', '')})

#adjust the number of folds as per user requirments
folds = [1]

#iterate through each fold
for fold in folds:
    #read each metadata file and make a dataframe out of this
    train= read_fold(f'fold{fold}_train.txt')
    validation = read_fold(f'fold{fold}_evaluate.txt')
    #concatenating labelled data into one dataframe
    train = pd.concat([train, validation], ignore_index=True)
    test= read_fold(f'fold{fold}_test.txt')  
    
    #iterate through each row of the dataframe with tqdm which provides us with a progress bar
    for row in tqdm(train.itertuples(), total=len(train)):
        # if the label folder doesnt exist, make that folder
        if not os.path.exists(f'{DATA_PATH}/dir/train/{row.scene}'):
            os.makedirs(f'{DATA_PATH}/dir/train/{row.scene}')
        #copy from the downloaded directory to the new directory
        shutil.copy2(f'{DATA_PATH}/audio/{row.file}', f'{DATA_PATH}/dir/train/{row.scene}')
    #iterate through each row of the dataframe with tqdm which provides us with a progress bar
    for row in tqdm(test.itertuples(), total=len(test)):
        # if the test folder doesnt exist, make that folder
        if not os.path.exists(f'{DATA_PATH}/dir/test'):
            os.makedirs(f'{DATA_PATH}/dir/test')
        #copy from downloaded directory to test folder
        shutil.copy2(f'{DATA_PATH}/audio/{row.file}', f'{DATA_PATH}/dir/test')
    

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
            
list_files(f'{DATA_PATH}/dir/train/')
    

      
