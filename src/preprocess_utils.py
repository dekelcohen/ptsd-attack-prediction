# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:48:13 2021

@author: Dekel
"""
#%% imports
import glob
import pandas as pd

class PreprocessUtils:
      
     def __init__(self,cfg):
          """
          Parameters
          ----------
          cfg : Config
               Pass config from caller.
          """
          self.cfg = cfg
          
          
     def merge_2_timeseries(self,df_l,df_r):
          return pd.merge_asof(df_l, df_r, on=self.cfg.TIMESTAMP_COL, tolerance=pd.Timedelta('60s'), direction='nearest')
     

     #%% Polar H10 files format preprocess utils 
     def preprocess_polar(self,df):
          # Convert local Phone datetime string --> UTC datetime64 
          ts =df['Phone timestamp'].astype('datetime64').dt.tz_localize('Israel').dt.tz_convert('UTC')     
          # calculate the sampling rate of the signal, using most accurate timestamps
          sampling_rate = int(1e+9 / df['sensor timestamp [ns]'].diff().quantile([.1,.25,.5,.75,.9,.95,.99]).mean())
          tm_cols = df.columns[df.columns.str.contains('timestamp')]
          df = df.drop(columns=tm_cols)
          df.insert(0,self.cfg.TIMESTAMP_COL, ts)
          return df, sampling_rate
     
     def read_polar_sensor_files(self,sig_type,polar_session_root_path):
          """
          Parameters
          ----------
          sig_type : string
               ECG, ACC, RR  - the sensor type to read recursively from the polar session folders. case sensitive
               must match part of the polar file name 
          polar_session_root_path : string
               root path of session folder (ex: XXX\data-large\first session)
               below there are folders with timestamps names:
               XXX\data-large\first session\2021-07-11T12_01-17_23\Polar_H10_8BF29B2E_20210711_120101_RR.txt
     
          Returns
          -------
          df of signal with timestamp - cleaned, normalized column names and types, concatenated from all session subfolders
     
          """
          dfiles = glob.glob(polar_session_root_path + f"/**/*{sig_type}.txt", recursive = True)
          dfs_sig = [pd.read_csv(file_path,sep=';') for file_path in dfiles]
          df_sig = pd.concat(dfs_sig)
          df_sig, sampling_rate = self.preprocess_polar(df_sig)
          df_sig.sort_values(by=[self.cfg.TIMESTAMP_COL],inplace=True)
          df_sig = df_sig.reset_index(drop=True)
          return df_sig, sampling_rate
          
     
     #%% Read preprocess labels file 
     def read_preprocess_labels(self,path):
          """
          read tags (labels) and return [tag,timestamp] df - UTC 
          """
          df_tags_raw = pd.read_csv(path,sep=',')
          df_tags = pd.DataFrame()
          df_tags[self.cfg.TIMESTAMP_COL] = df_tags_raw['updatedAt (S)'].astype('datetime64').dt.tz_localize('UTC')
          df_tags['tag'] = df_tags_raw['name (S)']
          df_tags.sort_values(by=[self.cfg.TIMESTAMP_COL],inplace=True)
          return df_tags
     
     #%% get indexes (0 based) of tags in signal, based on merge by timestamps between 2 dfs
     def get_event_indices_in_sig(self,df_sig,df_tags):
          df_tags_sig = self.merge_2_timeseries(df_tags,df_sig.reset_index().rename(columns={'index' : 'sig_index'}))
          tags_indices = df_tags_sig[~df_tags_sig['sig_index'].isna()]
          return tags_indices[[self.cfg.TIMESTAMP_COL,'tag','sig_index']].reset_index(drop=True)
     
     #%% Read raw data and merge it by time to a single df
     
     def load_merge_timeseries_data(self,acc_p_path,ecg_p_path):
          """
          load signals into dataframes (logs of sensors, indexed by timestamps)
          return  a merged df (by timestamp) - where eeach col is a sensor reading 
          """
          df_res = pd.read_csv(acc_p_path,sep=';')
          df_res = self.preprocess_polar(df_res)
          if ecg_p_path is not None:
               df_ecg_p = pd.read_csv(ecg_p_path,sep=';')
               df_ecg_p = self.preprocess_polar(df_ecg_p)
               df_res = self.merge_2_timeseries(df_res, df_ecg_p)
          return df_res
     
     
          