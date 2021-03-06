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
          self.per_sig_features = {
               'ECG' : self.get_feature_windows_for_ecg,
               'BVP' : self.get_feature_windows_for_ppg, # bvp == ppg
               }
          
     def get_feature_windows_for_ecg(self,epochs,sampling_rate):
          """
          Parameters
          ----------
          epochs : dict of df of signals, each df is ecg raw signal for a time win
          sampling_rate : int - Hz

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
     
     def get_feature_windows_for_ppg(self,epochs,sampling_rate):
          """
          Parameters
          ----------
          epochs : dict of df of signals, each df is ecg raw signal for a time win
          sampling_rate : int - Hz
               

          Returns
          -------
          df - row - features of a single tagged window (epoch) 
          each row has dozens of cols (features) -  Ex: ppg has 

          """
          # Perf: nk.ecg_process is very slow with large data --> must first cut to minimal windows and then process as little as possible 
          epochs_ppg_signals = [ nk.ppg_process(epoch['signal'], sampling_rate=sampling_rate) for epoch in epochs.values()]
          epochs_fts = [nk.ppg_intervalrelated(sig[0],sampling_rate) for sig in epochs_ppg_signals]
          df_epochs_fts = pd.concat(epochs_fts)
          return df_epochs_fts
     
     def gen_feature_windows_for_type(self,preut,sig_type,fmt,df_tags, window_start,window_end,add_n_windows_ba = 0,root_path=None):
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
          add_n_windows_ba - if not 0 - create <add_n_windows_ba> windows before and <n_windows_ba> after the tag
          Returns
          -------
          list of dfs of features of a sig type for all labeled windows
          each df has dozens of features (Ex: ecg has 58 - ECG_Rate_Mean  HRV_RMSSD  HRV_MeanNN ...)
     
          """
          if root_path is None:
               root_path = self.cfg.DATA_FOLDER
               
          sig_type = sig_type.upper()
          df_sig, sampling_rate = preut.read_sensor_files(sig_type,fmt,root_path)
          tags_indices = preut.get_event_indices_in_sig(df_sig,df_tags)
          ftr_indices = tags_indices # tagged idxs + optional added windows before and after (add_n_windows_ba)
          if add_n_windows_ba > 0:
               added_indices = preut.add_windows_indices(tags_indices=ftr_indices, df_sig=df_sig, n_windows=add_n_windows_ba, sampling_rate=sampling_rate, window_len = window_end - window_start)
               ftr_indices = pd.concat([ftr_indices,added_indices],axis=0)
               ftr_indices.sort_values(by=[self.cfg.TIMESTAMP_COL],inplace=True)
               ftr_indices = ftr_indices.reset_index(drop=True)
                
          evt_idxs = ftr_indices.sig_index.tolist() 
          epochs = nk.epochs_create(df_sig, events=evt_idxs, sampling_rate=sampling_rate, epochs_start=window_start, epochs_end=window_end)
          get_feature_windows_mtd = self.per_sig_features[sig_type]
          df_epochs_fts = get_feature_windows_mtd(epochs,sampling_rate)   
          features_cols = df_epochs_fts.columns.tolist()
          df_epochs_fts = pd.concat([df_epochs_fts.reset_index(drop=True),ftr_indices],axis=1)
          return df_epochs_fts, features_cols
     
          