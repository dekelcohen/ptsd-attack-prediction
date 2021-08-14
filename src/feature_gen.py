# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 00:32:49 2021

@author: Dekel
"""

# Change Working Directory: D:\Dekel\Data\PTSD\ptsd-attack-prediction\src
# %% imports

import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import os
from config import get_config
from preprocess_utils import PreprocessUtils
from features_utils import FeaturesUtils

DATA_FOLDER = '../../data-large/first session'
cfg = get_config(data_folder=DATA_FOLDER)
cfg.TAGS_PATH = os.path.join(cfg.DATA_FOLDER, 'first session tagging.csv')

# %% Samples
# See sample code at: https://github.com/neuropsychology/NeuroKit/blob/dev/docs/examples/intervalrelated.ipynb
# Event related (events are short <= 10 secs periods vs intervals - longer) https://colab.research.google.com/drive/1IAAzyeD8V3kam1uKDScwZjWaxbXg85ua#scrollTo=o7DyHfzuIWzM

# %% Constants
# Calc Polar  ecg sampling rate in Hz: 1e+9 / df_sig['sensor timestamp [ns]'].diff()
POLAR_H10_ECG_SAMPLING_RATE_HZ = 130


# %% Config
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size'] = 13

preut = PreprocessUtils(cfg=cfg)
featut = FeaturesUtils(cfg=cfg)

# %% Read GT (tags/labels) and cleanit
df_tags = preut.read_preprocess_labels(path=cfg.TAGS_PATH)

# %% Read data
# Read a single file type (ex: ECG) from all session subfolders

# %% Extract features
windows_fts = featut.gen_feature_windows_for_type(
    preut=preut, sig_type='ECG', df_tags=df_tags, window_start=0, window_end=5*60)


################################### Draft Test ###########################################
# %% Exract signal
ecg_signals, info = nk.ecg_process(
    df_sig['ecg [uV]'], sampling_rate=POLAR_H10_ECG_SAMPLING_RATE_HZ)
plot = nk.ecg_plot(ecg_signals[:3000],
                   sampling_rate=POLAR_H10_ECG_SAMPLING_RATE_HZ)

# %% Segment the signal to windows
# This returns a dictionary of 2 processed ECG dataframes, which you can then enter into ecg_intervalrelated().
# Half the data
epochs = nk.epochs_create(ecg_signals, events=[
                          0, 15000], sampling_rate=100, epochs_start=0, epochs_end=150)

# %% Extract ECG window features
df_fts = nk.ecg_intervalrelated(ecg_signals)


# Tests#######################################3
# Test: df_sig = read_polar_sensor_files('ECG',polar_session_root_path=cfg.DATA_FOLDER,cfg=cfg)
