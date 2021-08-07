# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:18:00 2021

@author: Dekel
"""
import os

class Config():
     pass

def get_config(data_folder):
     cfg = Config()
     cfg.DATA_FOLDER = data_folder
     cfg.POLAR_FOLDER = os.path.join(data_folder,'polar')
     cfg.POLAR_ECG_PATH =  os.path.join(cfg.POLAR_FOLDER, 'Polar_H10_55354125_20210516_080846_ECG.txt')
     cfg.POLAR_ACC_PATH =  os.path.join(cfg.POLAR_FOLDER, 'Polar_H10_55354125_20210516_080900_ACC.txt')
     cfg.TIMESTAMP_COL = 'timestamp'
     cfg.activity = Config()
     cfg.activity.window = Config()
     cfg.activity.window.LENGTH = 256
     cfg.activity.window.OVERLAP = 128
     cfg.activity.BATCH_SIZE = 3 # TODO:Debug:Restore to 32 / 64
     return cfg
