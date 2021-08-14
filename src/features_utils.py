# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 19:37:12 2021

@author: Dekel
"""
import pandas as pd
import neurokit2 as nk

class FeaturesUtils:
#%% Set config from caller to be used in this module - no need to pass cfg to every func call
     def __init__(self,cfg):
          self.cfg = cfg
          
     def get_feature_windows_for_ecg(self,epochs,sampling_rate):
          """
          Parameters
          ----------
          epochs : dict of df of signals, each df is ecg raw signal for a time win
          sampling_rate : TYPE
               DESCRIPTION.

          Returns
          -------
          df - row - features of a single tagged window (epoch) 
          each row has dozens of cols (features) -  Ex: ecg has 58 - ECG_Rate_Mean  HRV_RMSSD  HRV_MeanNN ... 

          """
          # Perf: nk.ecg_process is very slow with large data --> must first cut to minimal windows and then process as little as possible 
          epochs_ecg_signals = [ nk.ecg_process(epoch['ecg [uV]'].astype('int64'), sampling_rate=sampling_rate) for epoch in epochs.values()]
          epochs_fts = [nk.ecg_intervalrelated(sig[0],sampling_rate) for sig in epochs_ecg_signals]
          df_epochs_fts = pd.concat(epochs_fts)
          return df_epochs_fts
          
     def gen_feature_windows_for_type(self,preut,sig_type,df_tags, window_start,window_end):
          """
          Parameters
          ----------
          sig_type : string
               ECG, ACC, RR  - the sig type to generate features for
          df_tags: df
               dataframe of labels with timestamps, to merge_asof with the sesnor timestamps
          window_start,window_end: int (seconds)
               start and end of a time window around a label in seconds - used to compute features for the time window
               ex: window_start=-8 --> time window starts 8 seconds before the label 
               window_end=7 --> time window starts 7 seconds after the label 
          Returns
          -------
          list of dfs of features of a sig type for all labeled windows
          each df has dozens of features (Ex: ecg has 58 - ECG_Rate_Mean  HRV_RMSSD  HRV_MeanNN ...)
     
          """
          sig_type = sig_type.upper()
          df_sig, sampling_rate = preut.read_polar_sensor_files(sig_type,polar_session_root_path=self.cfg.DATA_FOLDER)
          tags_indices = preut.get_event_indices_in_sig(df_sig,df_tags)
          epochs = nk.epochs_create(df_sig, events=tags_indices.sig_index.tolist(), sampling_rate=sampling_rate, epochs_start=window_start, epochs_end=window_end)
          if sig_type =='ECG':
               df_epochs_fts = self.get_feature_windows_for_ecg(self,epochs,sampling_rate)   
          df_epochs_fts = pd.concat([df_epochs_fts.reset_index(drop=True),tags_indices],axis=1)
          return df_epochs_fts
     
          