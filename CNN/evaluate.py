# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 23:54:51 2021

@author: Sarvagya
Uses the sed eval package to write a file in dcase format 
"""
import sed_eval
import dcase_util
import pandas as pd

DATA_PATH = 'E:/Acoustic scene/DCASE2017-baseline-system-master/applications/data/TUT-acoustic-scenes-2017-development'
class accuracies:
    def __init__(self, fold, dir_path):
        self.fold=fold
        self.dir=dir_path
    
    def evaluate(self, estimated):
        def create_format():#this function takes the ground truth metadata and creates a list of dict as acceptable to sed eval
            data= pd.read_csv(f'{DATA_PATH}/evaluation_setup/fold{self.fold}_{self.dir}.txt', 
                               sep='\t', names=['file', 'scene'])
            ls= data.values.tolist()
            keys= ['scene_label', 'file']
            reference= []
        
            for i in range(len(ls)):
                reference.append(dict(zip(keys,[ls[i][1], ls[i][0]])))
            return reference
        
        
        
        reference= create_format() #call the above function
        
        reference = dcase_util.containers.MetaDataContainer(reference) #create ref metadata container

        estimated = dcase_util.containers.MetaDataContainer(estimated) #create estimated metadata container

        scene_labels = sed_eval.sound_event.util.unique_scene_labels(reference) #extract scene labels

        scene_metrics = sed_eval.scene.SceneClassificationMetrics(scene_labels) #use the labels for comparison between 2 files
        scene_metrics.evaluate(
            reference_scene_list=reference,
            estimated_scene_list=estimated
            )

        file= open(f'metrics/fold{self.fold}_{self.dir}_metrics.txt', 'wt') #write the classification metrics into this file
        file.write(str(scene_metrics))
        file.close()
        