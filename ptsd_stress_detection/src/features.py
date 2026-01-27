import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy import stats

class FeatureExtractor:
    """
    Extracts physiological features for stress detection.
    Focuses on robust statistical features (Mean/Max/Slope) over 10-minute windows,
    as recommended by JMIR 2023 for ambulatory data with potential signal loss.
    """
    
    def __init__(self, window_size_min=5):
        self.window_size_min = window_size_min
        # 5 minutes in seconds
        self.window_sec = window_size_min * 60

    def compute_hr_features(self, hr_monitor_data: np.ndarray) -> dict:
        """
        Computes Heart Rate features including advanced HRV metrics.
        Robust to sparse data (common in Empatica IBI).
        """
        if len(hr_monitor_data) == 0:
            return {
                "hr_mean": np.nan, "hr_min": np.nan, "hr_max": np.nan,
                "hr_std": np.nan, "hr_slope": np.nan, "hr_rmssd": np.nan,
                "hr_sdnn": np.nan, "hr_pnn50": np.nan, "hr_range": np.nan
            }

        features = {
            "hr_mean": np.mean(hr_monitor_data),
            "hr_min": np.min(hr_monitor_data),
            "hr_max": np.max(hr_monitor_data),
            "hr_std": np.std(hr_monitor_data),
            "hr_range": np.max(hr_monitor_data) - np.min(hr_monitor_data),  # NEW: HR Range
        }
        
        # Calculate slope (linear regression over the window)
        if len(hr_monitor_data) > 1:
            slope, _, _, _, _ = stats.linregress(np.arange(len(hr_monitor_data)), hr_monitor_data)
            features["hr_slope"] = slope
            
            # HRV Features from IBI approximation
            # Filter out zero/invalid HR values to prevent division by zero
            valid_hr = hr_monitor_data[hr_monitor_data > 0]
            if len(valid_hr) > 1:
                ibi_series = 60.0 / valid_hr  # Convert HR to IBI (seconds)
                ibi_ms = ibi_series * 1000    # Convert to milliseconds
                ibi_diffs = np.diff(ibi_ms)
                
                # RMSSD: Root Mean Square of Successive Differences
                features["hr_rmssd"] = np.sqrt(np.mean(ibi_diffs ** 2))
                
                # SDNN: Standard Deviation of NN intervals (NEW)
                features["hr_sdnn"] = np.std(ibi_ms)
                
                # pNN50: Proportion of successive differences > 50ms (NEW)
                nn50 = np.sum(np.abs(ibi_diffs) > 50)
                features["hr_pnn50"] = (nn50 / len(ibi_diffs)) * 100 if len(ibi_diffs) > 0 else 0
            else:
                features["hr_rmssd"] = np.nan
                features["hr_sdnn"] = np.nan
                features["hr_pnn50"] = np.nan
        else:
            features["hr_slope"] = 0.0
            features["hr_rmssd"] = 0.0
            features["hr_sdnn"] = 0.0
            features["hr_pnn50"] = 0.0
            
        return features

    def compute_eda_features(self, decomposed_eda: dict) -> dict:
        """
        Computes EDA features from Phasic and Tonic components.
        """
        phasic = decomposed_eda["phasic"]
        tonic = decomposed_eda["tonic"]
        cleaned = decomposed_eda["cleaned"]
        
        # Phasic Features (SCR - Skin Conductance Response)
        # Find peaks in phasic signal (SCRs)
        peaks, _ = find_peaks(phasic, height=0.01) # 0.01 uS threshold
        
        features = {
            # Tonic (Slow) Features
            "eda_tonic_mean": np.mean(tonic),
            "eda_tonic_std": np.std(tonic),
            "eda_tonic_min": np.min(tonic),
            "eda_tonic_max": np.max(tonic),
            
            # Phasic (Fast) Features
            "eda_phasic_mean": np.mean(phasic),
            "eda_phasic_max": np.max(phasic) if len(phasic) > 0 else 0,
            "eda_scr_count": len(peaks), # Number of responses (SOTA feature)
            "eda_scr_freq": len(peaks) / self.window_size_min, # Peaks per minute (Normalized)
            "eda_phasic_auc": np.trapz(phasic), # Area Under Curve (Total activation)
            
            # Global
            "eda_mean": np.mean(cleaned),
            "eda_slope": stats.linregress(np.arange(len(cleaned)), cleaned)[0] if len(cleaned) > 1 else 0
        }
        
        return features

    def compute_temp_features(self, temp_data: np.array) -> dict:
        if len(temp_data) == 0: return {}
        
        # Skin Temperature features
        features = {
            "temp_mean": np.mean(temp_data),
            "temp_min": np.min(temp_data),
            "temp_max": np.max(temp_data),
            "temp_std": np.std(temp_data)
        }
        
        # Slope
        if len(temp_data) > 1:
            slope, _, _, _, _ = stats.linregress(np.arange(len(temp_data)), temp_data)
            features["temp_slope"] = slope
        else:
            features["temp_slope"] = 0.0
            
        return features

    def compute_acc_features(self, acc_df: pd.DataFrame) -> dict:
        if acc_df.empty: return {}
        
        # Calculate Magnitude: sqrt(x^2 + y^2 + z^2)
        # Note: Empatica raw acc is usually int (e.g. -60 to 60 or similar range per g)
        # We process magnitude to be orientation invariant
        
        x = acc_df['acc_x'].values
        y = acc_df['acc_y'].values
        z = acc_df['acc_z'].values
        
        mag = np.sqrt(x**2 + y**2 + z**2)
        
        features = {
            "acc_mean": np.mean(mag), # General activity level
            "acc_std": np.std(mag),   # Variation in movement
            "acc_max": np.max(mag)
        }
        return features

    def compute_cross_modal_features(self, eda_features: dict, hr_features: dict) -> dict:
        """
        Computes cross-modal features combining EDA and HR signals.
        Stress often shows divergent patterns between modalities.
        """
        features = {}
        
        # EDA/HR Ratio - stress shows elevated EDA with elevated HR
        eda_mean = eda_features.get('eda_mean', np.nan)
        hr_mean = hr_features.get('hr_mean', np.nan)
        
        if not np.isnan(eda_mean) and not np.isnan(hr_mean) and hr_mean > 0:
            features['eda_hr_ratio'] = eda_mean / hr_mean * 100  # Scaled
        else:
            features['eda_hr_ratio'] = np.nan
            
        # SCR rate / HR ratio - sympathetic activation pattern
        scr_freq = eda_features.get('eda_scr_freq', 0)
        if not np.isnan(hr_mean) and hr_mean > 0:
            features['scr_hr_ratio'] = scr_freq / hr_mean * 1000
        else:
            features['scr_hr_ratio'] = np.nan
            
        # Phasic EDA * HR product - combined arousal indicator
        phasic_max = eda_features.get('eda_phasic_max', 0)
        if not np.isnan(hr_mean):
            features['arousal_product'] = phasic_max * hr_mean
        else:
            features['arousal_product'] = np.nan
            
        return features

    def compute_derivative_features(self, signal: np.ndarray, prefix: str) -> dict:
        """
        Computes rate-of-change features for a signal.
        Useful for detecting rapid stress onset.
        """
        features = {}
        
        if len(signal) < 2:
            features[f'{prefix}_deriv_mean'] = 0.0
            features[f'{prefix}_deriv_max'] = 0.0
            features[f'{prefix}_deriv_std'] = 0.0
            return features
            
        deriv = np.diff(signal)
        features[f'{prefix}_deriv_mean'] = np.mean(deriv)
        features[f'{prefix}_deriv_max'] = np.max(np.abs(deriv))
        features[f'{prefix}_deriv_std'] = np.std(deriv)
        
        return features

    def create_windows(self, df_eda: pd.DataFrame, df_hr: pd.DataFrame) -> list:
        """
        Segment data into windows. 
        Assumes dataframes have a 'timestamp' column.
        Returns a list of dictionaries with valid window features.
        """
        # Handle empty DataFrames
        if df_eda.empty and df_hr.empty:
            return []
        
        # Align range - handle case where one DataFrame might be empty
        timestamps = []
        if not df_eda.empty:
            timestamps.append(df_eda['timestamp'].min())
            timestamps.append(df_eda['timestamp'].max())
        if not df_hr.empty:
            timestamps.append(df_hr['timestamp'].min())
            timestamps.append(df_hr['timestamp'].max())
        
        start_time = min(timestamps)
        end_time = max(timestamps)
        
        windows = []
        current_time = start_time
        
        while current_time + self.window_sec <= end_time:
            window_end = current_time + self.window_sec
            
            # Slice data
            eda_slice = df_eda[(df_eda['timestamp'] >= current_time) & (df_eda['timestamp'] < window_end)]['eda'].values
            hr_slice = df_hr[(df_hr['timestamp'] >= current_time) & (df_hr['timestamp'] < window_end)]['hr'].values
            
            # Process window if sufficient data exists
            if len(eda_slice) > 0 and len(hr_slice) > 0:
                windows.append({
                    "start_time": current_time,
                    "end_time": window_end,
                    "eda_raw": eda_slice,
                    "hr_raw": hr_slice
                })
            
            # Sliding window (no overlap logic here, can add step)
            current_time += self.window_sec
            
        return windows
