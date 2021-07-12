# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 00:32:49 2021

@author: Dekel
"""

#%% imports

import pandas as pd
import matplotlib.pyplot as plt
import neurokit2 as nk
import os

#%% Samples
# See sample code at: https://github.com/neuropsychology/NeuroKit/blob/dev/docs/examples/intervalrelated.ipynb
# Event related (events are short <= 10 secs periods vs intervals - longer) https://colab.research.google.com/drive/1IAAzyeD8V3kam1uKDScwZjWaxbXg85ua#scrollTo=o7DyHfzuIWzM

#%% Constants
# Calc Polar  ecg sampling rate in Hz: 1e+9 / df_sig['sensor timestamp [ns]'].diff()
POLAR_H10_ECG_SAMPLING_RATE_HZ = 130

dirname = os.path.dirname(__file__)

#%% Config
plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images
plt.rcParams['font.size']= 13

#%% Read data
df_sig = pd.read_csv('../data/polar/Polar_H10_55354125_20210516_080846_ECG.txt',sep=';')

#%% Exract signal
ecg_signals, info = nk.ecg_process(df_sig['ecg [uV]'], sampling_rate=POLAR_H10_ECG_SAMPLING_RATE_HZ)
plot = nk.ecg_plot(ecg_signals[:3000], sampling_rate=POLAR_H10_ECG_SAMPLING_RATE_HZ)

#%% Segment the signal to windows 
# This returns a dictionary of 2 processed ECG dataframes, which you can then enter into ecg_intervalrelated().
# Half the data
epochs = nk.epochs_create(ecg_signals, events=[0, 15000], sampling_rate=100, epochs_start=0, epochs_end=150)

#%% Extract ECG window features 
df_fts = nk.ecg_intervalrelated(ecg_signals)
