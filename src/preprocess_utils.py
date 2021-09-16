# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 00:48:13 2021

@author: Dekel
"""
#%% imports
import glob
from datetime import datetime 
from itertools import chain
from statistics import mean,stdev
import pandas as pd
class PreprocessUtils:
      
     def __init__(self,cfg):
          """
          Parameters
          ----------
          cfg : Config
               Pass config from caller.
          """
          # preprocessing variants for polar, empatica and other sensor file formats 
          self.file_formats = {
               'polar_csv' : { 'header' : 'infer','sep' : ';', 'sig_type_globexpr': '/**/*{sig_type}.txt' , 'preprocess_mtd' :  self.preprocess_polar },
               'empatica_csv' : { 'header' : None, 'sep' : ' ', 'sig_type_globexpr': '/**/*{sig_type}.csv' , 'preprocess_mtd' :  self.preprocess_empatica }
               }

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
     
     def preprocess_empatica(self,df):
          df = df.rename(columns={0 : 'signal'})
          start_unix_timestamp = df.iloc[0][0]
          start_dt_utc = datetime.utcfromtimestamp(start_unix_timestamp)
          sampling_rate = int(df.iloc[1][0])
          df = df.iloc[2:]
          ts = pd.date_range(start=start_dt_utc,periods=len(df),freq=f'{1000/sampling_rate}ms')
          df.insert(0,self.cfg.TIMESTAMP_COL, ts.tz_localize('UTC'))
          return df, sampling_rate
          
     def read_sensor_files(self,sig_type,fmt,root_path):
          """
          Parameters
          ----------
          sig_type : string
               ECG, ACC, RR  - the sensor type to read recursively from the polar session folders. case sensitive
               must match part of the polar file name 
          root_path : string
               root path of session folder (ex: XXX\data-large\first session)
               below there are folders with timestamps names:
               XXX\data-large\first session\2021-07-11T12_01-17_23\Polar_H10_8BF29B2E_20210711_120101_RR.txt
     
          Returns
          -------
          df of signal with timestamp - cleaned, normalized column names and types, concatenated from all session subfolders
     
          """
          fmt_cfg = self.file_formats[fmt]
          header = fmt_cfg['header']
          sep = fmt_cfg['sep']
          sig_type_globexpr = fmt_cfg['sig_type_globexpr'].format(sig_type = sig_type)
          preprocess_mtd = fmt_cfg['preprocess_mtd']
          dfiles = glob.glob(root_path + sig_type_globexpr, recursive = True)
          lst_tup = [preprocess_mtd(pd.read_csv(file_path,header=header,sep=sep)) for file_path in dfiles]
          sigs_samplings = list(zip(*lst_tup))
          dfs_sig = sigs_samplings[0]
          sampling_rate = mean(sigs_samplings[1])
          if stdev(sigs_samplings[1]) > 0:
               raise Exception(f'Error: sampling rate is different in {sig_type} files: {sigs_samplings[1]}')
          #df_sig, sampling_rate = preprocess_mtd(df_sig)
          df_sig = pd.concat(dfs_sig)
          
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
          df_tags[self.cfg.TIMESTAMP_COL] = df_tags_raw['startLocalTime'].astype('datetime64').dt.tz_localize('UTC')
          df_tags['endUtcTime'] = df_tags_raw['endLocalTime'].astype('datetime64').dt.tz_localize('UTC')
          df_tags['tag'] = df_tags_raw['name']
          df_tags['userName'] = df_tags_raw['userName']
          df_tags.sort_values(by=[self.cfg.TIMESTAMP_COL],inplace=True)
          return df_tags
     
     #%% get indexes (0 based) of tags in signal, based on merge by timestamps between 2 dfs
     def get_event_indices_in_sig(self,df_sig,df_tags):
          df_tags_sig = self.merge_2_timeseries(df_tags,df_sig.reset_index().rename(columns={'index' : 'sig_index'}))
          tags_indices = df_tags_sig[~df_tags_sig['sig_index'].isna()]
          return tags_indices[[self.cfg.TIMESTAMP_COL,'tag','sig_index']].reset_index(drop=True)
     
     #%% Create windows before/after tags  
     def add_windows_indices(tags_sig_index,df_sig, n_windows, sampling_rate, window_len):
          wnd_len_ind = window_len * sampling_rate
          def create_before_after_indxs(ind):
               before_idxs = range(ind,max(df_sig.index.min(), ind - n_windows*wnd_len_ind), -wnd_len_ind)
               after_idxs = range(ind,min(df_sig.index.max(), ind + n_windows*wnd_len_ind), wnd_len_ind)        
               return chain(before_idxs,after_idxs) 
               
          tags_sig_index = tags_sig_index.astype('int64')
          series_arrays = tags_sig_index.apply(create_before_after_indxs)
          added_indices = list(chain.from_iterable(series_arrays))
          return added_indices
          
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
     
     
          