import numpy as np
import pandas as pd
from scipy.signal import ellip, filtfilt, butter

class SignalProcessor:
    """
    Implements SOTA signal processing techniques for Empatica E4 data.
    Based on JMIR 2023 (Verma et al.) and PMC 2024 (Lazarou et al.).
    """
    
    def __init__(self, sampling_rate_eda=4, sampling_rate_hr=1):
        self.fs_eda = sampling_rate_eda
        self.fs_hr = sampling_rate_hr

    def remove_artifacts(self, data: np.ndarray, threshold_sigma: float = 3.0) -> np.ndarray:
        """
        Simple despiking: Replace values > threshold_sigma * std with interpolation.
        Used to remove sudden motion artifacts.
        """
        signal = data.copy()
        mean = np.mean(signal)
        std = np.std(signal)
        
        # Identify spikes
        spikes = np.abs(signal - mean) > threshold_sigma * std
        
        # Linear interpolation for spikes
        if np.sum(spikes) > 0:
            x_range = np.arange(len(signal))
            signal[spikes] = np.interp(x_range[spikes], x_range[~spikes], signal[~spikes])
            
        return signal

    def apply_elliptic_filter(self, data: np.ndarray, cutoff: float = 1.0) -> np.ndarray:
        """
        Applies an Elliptic Low-pass filter (SOTA for EDA).
        Standard cutoff for Empatica EDA is often cited around 0.8 - 1.1 Hz.
        """
        # Design Elliptic filter:
        # Passband ripple: 0.05 dB, Stopband attenuation: 40 dB
        # Order is typically determined by requirements, using 4th order as robust default
        try:
            b, a = ellip(N=4, rp=0.05, rs=40, Wn=cutoff, btype='low', fs=self.fs_eda)
            filtered_data = filtfilt(b, a, data)
            return filtered_data
        except Exception as e:
            print(f"Warning: Filter failed ({e}), returning raw data.")
            return data

    def decompose_eda(self, eda_signal: np.ndarray) -> dict:
        """
        Decomposes EDA into Phasic (Skin Conductance Response) and Tonic (Background) components.
        
        Uses a standard Butterworth low-pass filter to estimate the Tonic component (slow moving),
        subtracting it to get the Phasic component. 
        Note: While cvxEDA is gold standard, this is a robust scipy-only approximation 
        often used when convex optimization libraries are unavailable.
        """
        # 1. Clean data
        clean_eda = self.remove_artifacts(eda_signal)
        clean_eda = self.apply_elliptic_filter(clean_eda, cutoff=1.0)
        
        # 2. Extract Tonic (Very low frequency component, < 0.05 Hz)
        # Using a low-pass butterworth for tonic estimation
        b, a = butter(N=2, Wn=0.05, btype='low', fs=self.fs_eda)
        tonic = filtfilt(b, a, clean_eda)
        
        # 3. Extract Phasic (Residual)
        phasic = clean_eda - tonic
        
        # Ensure non-negativity for phasic (physics constraint)
        phasic = np.maximum(phasic, 0)
        
        return {
            "original": eda_signal,
            "cleaned": clean_eda,
            "tonic": tonic,
            "phasic": phasic
        }
