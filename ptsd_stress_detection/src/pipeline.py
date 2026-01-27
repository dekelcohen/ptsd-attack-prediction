import pandas as pd
import numpy as np
import os
import glob
import fastavro
from preprocessing import SignalProcessor
from features import FeatureExtractor
from model import StressClassifier

import pickle
from joblib import Parallel, delayed

class StressDetectionPipeline:
    def __init__(self, use_model="xgboost", cache_dir="cache", n_jobs=-1, 
                 window_size_min=10, focal_gamma=2.0):
        self.processor = SignalProcessor()
        self.extractor = FeatureExtractor(window_size_min=window_size_min)
        self.classifier = StressClassifier(model_type=use_model, focal_gamma=focal_gamma)
        self.cache_dir = cache_dir
        self.n_jobs = n_jobs
        self.window_size_min = window_size_min
        if self.cache_dir:
            self.cache_dir = os.path.abspath(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
        self.invalid_timestamp_ranges = set()
        self.invalid_starts = np.array([])
        self.invalid_ends = np.array([])
        print(f"Pipeline Initialized. Cache Dir: '{self.cache_dir}', n_jobs={n_jobs}")
        
    def find_avro_files(self, root_dir: str):
        avro_files = glob.glob(os.path.join(root_dir, "**", "*.avro"), recursive=True)
        print(f"Found {len(avro_files)} Avro files.")
        return avro_files
    
    def process_files_parallel(self, files: list, overlap_percent=0.75) -> pd.DataFrame:
        """
        Process multiple AVRO files in parallel using joblib.
        Returns concatenated features DataFrame.
        """
        print(f"Processing {len(files)} files in parallel (n_jobs={self.n_jobs})...")
        
        def process_single(f):
            try:
                res = self.process_avro_file(f)
                if res:
                    df_eda, df_hr, df_temp, df_acc = res
                    feats = self.create_features_from_streams(df_eda, df_hr, df_temp, df_acc, overlap_percent=overlap_percent)
                    if not feats.empty:
                        feats['source_file'] = os.path.basename(f)
                        return feats
            except Exception as e:
                pass  # Silent fail for parallel
            return None
        
        results = Parallel(n_jobs=self.n_jobs, verbose=5)(
            delayed(process_single)(f) for f in files
        )
        
        valid_results = [r for r in results if r is not None]
        if valid_results:
            return pd.concat(valid_results, ignore_index=True)
        return pd.DataFrame()
    
    def process_participant_cached(self, data_dir: str, participant_id: str, 
                                    overlap_percent=0.75) -> pd.DataFrame:
        """
        Process a participant's files with feature caching.
        Loads from cache if available, otherwise processes and saves.
        """
        cache_path = os.path.join(self.cache_dir, f"features_{participant_id}.pkl") if self.cache_dir else None
        
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading features from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                print(f"Loaded {len(features)} feature windows from cache.")
                return features
            except Exception as e:
                print(f"Cache load failed: {e}. Reprocessing...")
        
        # Load invalid timestamps for this participant
        self.load_invalid_timestamps_from_aggregated(data_dir, participant_id=participant_id)
        
        # Find and process files
        all_files = self.find_avro_files(data_dir)
        participant_files = [f for f in all_files if participant_id in f]
        print(f"Found {len(participant_files)} files for {participant_id}")
        
        if not participant_files:
            return pd.DataFrame()
        
        # Use parallel processing
        features = self.process_files_parallel(participant_files, overlap_percent=overlap_percent)
        
        # Save to cache
        if cache_path and not features.empty:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(features, f)
                print(f"Saved features to cache: {cache_path}")
            except Exception as e:
                print(f"Cache save failed: {e}")
        
        return features

    def load_invalid_timestamps_from_aggregated(self, root_dir: str, participant_id: str = None):
        """
        Load invalid minute timestamps from aggregated_per_minute CSV files.
        Checks cache first. If not found, scans files and caches result.
        
        Args:
            root_dir: Directory containing participant data
            participant_id: Optional. If provided, filters specifically for this participant 
                           and uses a specific cache file.
        """
        cache_filename = f"invalid_timestamps_{participant_id}.pkl" if participant_id else "invalid_timestamps_all.pkl"
        cache_path = os.path.join(self.cache_dir, cache_filename) if self.cache_dir else None
        
        # Try loading from cache
        if cache_path and os.path.exists(cache_path):
            try:
                print(f"Loading invalid timestamps from cache: {cache_path}")
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                    self.invalid_timestamp_ranges = data['ranges']
                    self.invalid_starts = data['starts']
                    self.invalid_ends = data['ends']
                print(f"Loaded {len(self.invalid_timestamp_ranges)} invalid minute ranges from cache.")
                return self.invalid_timestamp_ranges
            except Exception as e:
                print(f"Failed to load cache: {e}. Reprocessing...")

        invalid_ranges = set()
        
        # Find aggregated per minute CSV files (look for eda.csv as reference)
        # Search pattern depends on whether participant_id is provided
        if participant_id:
            print(f"Scanning for invalid timestamps for participant: {participant_id}")
            # Assuming structure: root_dir / participant_id / ...
            # or simply filtering results. Let's filter glob results for safety.
            csv_pattern = os.path.join(root_dir, "**", "aggregated_per_minute", "*_eda.csv")
            all_files = glob.glob(csv_pattern, recursive=True)
            csv_files = [f for f in all_files if participant_id in f]
        else:
            print("Scanning for invalid timestamps (ALL participants)...")
            csv_pattern = os.path.join(root_dir, "**", "aggregated_per_minute", "*_eda.csv")
            csv_files = glob.glob(csv_pattern, recursive=True)
        
        print(f"Found {len(csv_files)} aggregated CSV files for invalid timestamp detection.")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                
                # Check if missing_value_reason column exists
                if 'missing_value_reason' not in df.columns:
                    continue
                
                # Get timestamp column (usually timestamp_unix in milliseconds)
                ts_col = None
                for col in ['timestamp_unix', 'timestamp', 'unix_timestamp']:
                    if col in df.columns:
                        ts_col = col
                        break
                
                if ts_col is None:
                    continue
                
                # Filter rows where missing_value_reason has a value (not NaN/empty)
                invalid_mask = df['missing_value_reason'].notna() & (df['missing_value_reason'] != '')
                invalid_rows = df[invalid_mask]
                
                for _, row in invalid_rows.iterrows():
                    ts = row[ts_col]
                    # Convert to seconds if in milliseconds
                    if ts > 1e12:  # Likely milliseconds
                        ts = ts / 1000
                    # Each row represents a 1-minute interval
                    minute_start = int(ts)
                    minute_end = minute_start + 60
                    invalid_ranges.add((minute_start, minute_end))
                    
            except Exception as e:
                print(f"Warning: Could not process {csv_file}: {e}")
        
        # Prepare optimized search structures
        if invalid_ranges:
            sorted_ranges = sorted(list(invalid_ranges))
            self.invalid_starts = np.array([r[0] for r in sorted_ranges])
            self.invalid_ends = np.array([r[1] for r in sorted_ranges])
        else:
            self.invalid_starts = np.array([])
            self.invalid_ends = np.array([])

        self.invalid_timestamp_ranges = invalid_ranges
        print(f"Loaded {len(invalid_ranges)} invalid minute ranges to exclude.")
        
        # Save to cache
        if cache_path:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump({
                        'ranges': self.invalid_timestamp_ranges,
                        'starts': self.invalid_starts,
                        'ends': self.invalid_ends
                    }, f)
                print(f"Saved invalid timestamps to cache: {cache_path}")
            except Exception as e:
                print(f"Failed to save cache: {e}")
                
        return invalid_ranges
    
    def is_timestamp_valid(self, timestamp_sec: float) -> bool:
        """
        Check if a timestamp falls within any invalid minute range.
        Uses binary search for O(log M) performance.
        """
        if self.invalid_starts.size == 0:
            return True
        
        # Find position where timestamp would fit
        idx = np.searchsorted(self.invalid_starts, timestamp_sec, side='right') - 1
        if idx < 0:
            return True
            
        return not (self.invalid_starts[idx] <= timestamp_sec < self.invalid_ends[idx])

    def process_avro_file(self, file_path: str):
        try:
            with open(file_path, 'rb') as f:
                reader = fastavro.reader(f)
                eda_segments = []
                hr_segments = []
                temp_segments = []
                acc_segments = []
                
                for record in reader:
                    raw = record.get('rawData', {})
                    
                    # 1. EDA
                    eda_data = raw.get('eda', {})
                    if eda_data and len(eda_data.get('values', [])) > 0:
                        start = eda_data.get('timestampStart')
                        fs = eda_data.get('samplingFrequency')
                        
                        # Validate Metadata
                        if start is None or fs is None or fs <= 0:
                            continue
                            
                        start = start / 1e6 
                        vals = np.array(eda_data['values'])
                        times = start + np.arange(len(vals)) / fs
                        
                        decomposed = self.processor.decompose_eda(vals)
                        
                        eda_df = pd.DataFrame({
                            'timestamp': times,
                            'eda_raw': vals,
                            'eda_clean': decomposed['cleaned'],
                            'eda_phasic': decomposed['phasic'],
                            'eda_tonic': decomposed['tonic']
                        })
                        eda_segments.append(eda_df)
                    
                    # 2. HR
                    peaks_data = raw.get('systolicPeaks', {})
                    peaks_nanos = peaks_data.get('peaksTimeNanos', [])
                    if len(peaks_nanos) > 1:
                        peaks_sec = np.array(peaks_nanos) / 1e9
                        ibi = np.diff(peaks_sec)
                        
                        # Avoid div by zero in IBI
                        valid_ibi = ibi > 0.001
                        if np.sum(valid_ibi) < 1: continue
                        
                        hr_vals = 60.0 / ibi[valid_ibi]
                        hr_times = peaks_sec[1:][valid_ibi]
                        
                        valid_mask = (hr_vals > 30) & (hr_vals < 200)
                        hr_df = pd.DataFrame({'timestamp': hr_times[valid_mask], 'hr': hr_vals[valid_mask]})
                        hr_segments.append(hr_df)
                        
                    # 3. Temperature
                    temp_data = raw.get('temperature', {})
                    if temp_data and len(temp_data.get('values', [])) > 0:
                        start = temp_data['timestampStart'] / 1e6
                        fs = temp_data['samplingFrequency']
                        vals = np.array(temp_data['values'])
                        times = start + np.arange(len(vals)) / fs
                        
                        temp_df = pd.DataFrame({'timestamp': times, 'temp': vals})
                        temp_segments.append(temp_df)
                        
                    # 4. Accelerometer
                    acc_data = raw.get('accelerometer', {})
                    if acc_data and len(acc_data.get('x', [])) > 0:
                        start = acc_data['timestampStart'] / 1e6
                        fs = acc_data['samplingFrequency']
                        times = start + np.arange(len(acc_data['x'])) / fs
                        
                        acc_df = pd.DataFrame({
                            'timestamp': times,
                            'acc_x': acc_data['x'],
                            'acc_y': acc_data['y'],
                            'acc_z': acc_data['z']
                        })
                        acc_segments.append(acc_df)
                
                # Combine
                if not eda_segments: return None # Min requirement
                
                full_eda = pd.concat(eda_segments).sort_values('timestamp')
                full_hr = pd.concat(hr_segments).sort_values('timestamp') if hr_segments else pd.DataFrame()
                full_temp = pd.concat(temp_segments).sort_values('timestamp') if temp_segments else pd.DataFrame()
                full_acc = pd.concat(acc_segments).sort_values('timestamp') if acc_segments else pd.DataFrame()
                
                return full_eda, full_hr, full_temp, full_acc
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def create_features_from_streams(self, df_eda, df_hr, df_temp, df_acc, overlap_percent=0.5):
        all_features = []
        if df_eda.empty: return pd.DataFrame()
        
        # Filter out invalid timestamps from all dataframes
        if self.invalid_timestamp_ranges:
            df_eda = self._filter_invalid_timestamps(df_eda)
            if not df_hr.empty:
                df_hr = self._filter_invalid_timestamps(df_hr)
            if not df_temp.empty:
                df_temp = self._filter_invalid_timestamps(df_temp)
            if not df_acc.empty:
                df_acc = self._filter_invalid_timestamps(df_acc)
        
        if df_eda.empty: return pd.DataFrame()
        
        # Determine range
        frames = [df for df in [df_eda, df_hr, df_temp, df_acc] if not df.empty]
        start_time = min(df['timestamp'].min() for df in frames)
        end_time = max(df['timestamp'].max() for df in frames)
        window_sec = self.extractor.window_sec
        # Step size based on overlap
        step_sec = int(window_sec * (1 - overlap_percent))
        if step_sec <= 0: step_sec = window_sec # Safety
        
        current_time = start_time
        while current_time + window_sec <= end_time:
            window_end = current_time + window_sec
            
            # Slice each stream
            eda_win = df_eda[(df_eda['timestamp'] >= current_time) & (df_eda['timestamp'] < window_end)]
            hr_win = df_hr[(df_hr['timestamp'] >= current_time) & (df_hr['timestamp'] < window_end)] if not df_hr.empty else pd.DataFrame()
            temp_win = df_temp[(df_temp['timestamp'] >= current_time) & (df_temp['timestamp'] < window_end)] if not df_temp.empty else pd.DataFrame()
            acc_win = df_acc[(df_acc['timestamp'] >= current_time) & (df_acc['timestamp'] < window_end)] if not df_acc.empty else pd.DataFrame()
            
            if len(eda_win) > 10: # Only EDA is strictly required? Or should enforce all?
                # Prepare EDA
                eda_prep = {
                    "cleaned": eda_win['eda_clean'].values,
                    "phasic": eda_win['eda_phasic'].values,
                    "tonic": eda_win['eda_tonic'].values
                }
                eda_feats = self.extractor.compute_eda_features(eda_prep)
                
                # HR
                hr_feats = self.extractor.compute_hr_features(hr_win['hr'].values) if not hr_win.empty else {}
                
                # Temp
                temp_feats = self.extractor.compute_temp_features(temp_win['temp'].values) if not temp_win.empty else {}
                
                # Acc
                acc_feats = self.extractor.compute_acc_features(acc_win) if not acc_win.empty else {}
                
                # Cross-modal features (EDA/HR ratios)
                cross_feats = self.extractor.compute_cross_modal_features(eda_feats, hr_feats)
                
                # Derivative features (rate of change)
                eda_deriv = self.extractor.compute_derivative_features(eda_win['eda_clean'].values, 'eda')
                hr_deriv = self.extractor.compute_derivative_features(hr_win['hr'].values, 'hr') if not hr_win.empty else {}
                
                combined = {**eda_feats, **hr_feats, **temp_feats, **acc_feats, **cross_feats, **eda_deriv, **hr_deriv}
                combined['start_time'] = current_time
                combined['end_time'] = window_end
                
                # Context Features
                # Time-aware features (Israel Time)
                ts_dt = pd.to_datetime(current_time, unit='s', utc=True)
                ts_israel = ts_dt.tz_convert('Israel')
                
                # Basic time features
                combined['hour_of_day'] = ts_israel.hour
                combined['day_of_week'] = ts_israel.dayofweek  # 0=Monday, 6=Sunday
                combined['is_weekend'] = 1 if ts_israel.dayofweek >= 5 else 0
                
                # Cyclical encodings (helps model understand 23:00 is close to 00:00)
                hour_rad = 2 * np.pi * ts_israel.hour / 24
                combined['hour_sin'] = np.sin(hour_rad)
                combined['hour_cos'] = np.cos(hour_rad)
                
                day_rad = 2 * np.pi * ts_israel.dayofweek / 7
                combined['day_sin'] = np.sin(day_rad)
                combined['day_cos'] = np.cos(day_rad)
                
                all_features.append(combined)
            current_time += step_sec
            
        return pd.DataFrame(all_features)
    
    def _filter_invalid_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows from dataframe where timestamp falls within invalid ranges.
        Optimized version using binary search: O(N log M).
        """
        if 'timestamp' not in df.columns or self.invalid_starts.size == 0:
            return df
        
        # Convert to numpy for vectorized operations
        timestamps = df['timestamp'].values
        
        # For each timestamp, find the index of the closest preceding invalid range start
        # np.searchsorted(side='right') - 1 gives us the largest i such that starts[i] <= timestamp
        indices = np.searchsorted(self.invalid_starts, timestamps, side='right') - 1
        
        # Check if the timestamp actually falls inside that specific range
        # Note: indices can be -1 if timestamp < any start
        in_range = (indices >= 0) & (timestamps < self.invalid_ends[np.maximum(0, indices)])
        
        valid_mask = ~in_range
        filtered_df = df[valid_mask].copy()
        
        rows_removed = len(df) - len(filtered_df)
        if rows_removed > 0:
            print(f"  Filtered out {rows_removed} rows with invalid timestamps")
        
        return filtered_df

    def load_labels(self, tags_file: str):
        try:
            df = pd.read_csv(tags_file)
            
            # Select Timestamp Column
            # STRICT RULE: Modified tags MUST use 'new_timestamp'
            if 'modified' in tags_file.lower():
                if 'new_timestamp' not in df.columns:
                    raise ValueError(f"Modified tags file '{tags_file}' missing required 'new_timestamp' column!")
                ts_col = 'new_timestamp'
                print(f"Loading MODIFIED tags from column: {ts_col}")
            else:
                ts_col = 'new_timestamp' if 'new_timestamp' in df.columns else 'timestamp'
                print(f"Loading tags from column: {ts_col}")
            
            # Standardize timestamp content
            # Strategy: Convert string -> localized datetime (Israel) -> unix seconds (UTC)
            try:
                # Clean up IDT/IST suffixes which confuse pd.to_datetime
                if df[ts_col].dtype == object:
                     clean_col = df[ts_col].astype(str).str.replace(' IDT', '', regex=False).str.replace(' IST', '', regex=False)
                else:
                     clean_col = df[ts_col]

                # Assuming format like "2025-07-31 13:02:00" is Israel Local Time
                # First convert to datetime (naive or mixed)
                # Use format='mixed' to handle both ISO8601 and simple strings
                df['dt'] = pd.to_datetime(clean_col, format='mixed')
                
                # Helper to standardize to Israel Time
                def localize_to_israel(ts):
                    if pd.isna(ts): return pd.NaT
                    if ts.tzinfo is None:
                        # Naive -> Localize to Israel
                        return ts.tz_localize('Israel', ambiguous='NaT', nonexistent='shift_forward')
                    else:
                        # Aware -> Convert to Israel
                        return ts.tz_convert('Israel')

                # Apply standardization (acceptable performance for tags which are few)
                df['dt'] = df['dt'].apply(localize_to_israel)
                    
                # Convert to Unix Seconds (UTC) to match feature extraction
                df['timestamp'] = df['dt'].astype('int64') // 10**9
                
            except Exception as e:
                print(f"Timestamp parsing failed: {e}")
                # Fallback to existing logic if simple parse fails
                try:
                    df['timestamp'] = pd.to_datetime(df[ts_col], utc=True).astype('int64') // 10**9
                except ValueError:
                    df['timestamp'] = df[ts_col] / 1000.0 # simple unix check
                
            stress_types = ['anxiety', 'anger', 'stress', 'fear', 'panic'] 
            df_stress = df[df['eventType'].str.lower().isin(stress_types)].copy()
            return df_stress
        except Exception as e:
            print(f"Error loading tags: {e}")
            return None

    def align_labels(self, features_df, tags_df, buffer_before_min=10, buffer_after_min=20):
        if tags_df is None or tags_df.empty:
            features_df['label'] = 0 
            return features_df

        features_df['label'] = 0
        # buffer_sec = buffer_min * 60  <-- Removed undefined/unused
        
        # Sort for speed (optional)
        tags_sorted = tags_df.sort_values('timestamp')
        
        for _, tag in tags_sorted.iterrows():
            t = tag['timestamp']
            start_range = t - (buffer_before_min * 60)
            end_range = t + (buffer_after_min * 60)
            
            mask = (features_df['end_time'] >= start_range) & (features_df['start_time'] <= end_range)
            features_df.loc[mask, 'label'] = 1
            
        return features_df

    def run_full_pipeline(self, data_dir: str, tags_file: str):
        print("--- Extracting Features ---")
        features = self.run_on_all_data(data_dir)
        if features is None or features.empty: 
            print("No features extracted.")
            return None
        
        print("--- Loading Labels ---")
        tags = self.load_labels(tags_file)
        print(f"Found {len(tags) if tags is not None else 0} stress tags.")
        
        print("--- Aligning Labels ---")
        labeled_data = self.align_labels(features, tags)
        print(f"Class Balance: {labeled_data['label'].value_counts().to_dict()}")
        
        print("--- Training Model ---")
        X = labeled_data.drop(columns=['label', 'start_time', 'end_time', 'source_file', 'timestamp'], errors='ignore')
        y = labeled_data['label']
        
        if len(y.unique()) < 2:
            print("Error: Only one class present (likely no stress tags matched). Cannot train.")
            return labeled_data
            
        self.classifier.train(X, y)
        self.classifier.evaluate(X, y) 
        
        return labeled_data

    def process_files(self, files: list):
        all_data = []
        for i, f in enumerate(files):
            # print(f"[{i+1}/{len(files)}] Processing {os.path.basename(f)}...")
            if (i+1) % 100 == 0: print(f"[{i+1}/{len(files)}] Processing...")
            
            # Check cache
            cache_path = None
            if self.cache_dir:
                filename = os.path.basename(f)
                cache_name = f"{filename}_features.pkl"
                cache_path = os.path.join(self.cache_dir, cache_name)
                # print(f"DEBUG: Checking {cache_path}")
                
                if os.path.exists(cache_path):
                    try:
                        feats = pd.read_pickle(cache_path)
                        # if not feats.empty: # Only if needed
                        if not feats.empty:
                            all_data.append(feats)
                        # print(f"Loaded from cache: {cache_path}")
                        continue
                    except Exception as e:
                        print(f"Cache load failed: {e}")
            
            # Process if not cached
            try:
                res = self.process_avro_file(f)
                if res:
                    df_eda, df_hr, df_temp, df_acc = res # Unpack 4 streams
                    feats = self.create_features_from_streams(df_eda, df_hr, df_temp, df_acc, overlap_percent=0.5)
                    if not feats.empty:
                        feats['source_file'] = os.path.basename(f)
                        all_data.append(feats)
                        
                        # Save to cache
                        # print(f"DEBUG: Attempting save with path: {cache_path}")
                        if cache_path:
                            try:
                                feats.to_pickle(cache_path)
                                # print(f"Saved: {os.path.basename(cache_path)}")
                            except Exception as e:
                                print(f"Cache save failed: {e}")
            except Exception as e:
                print(f"File processing failed: {e}")
                
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return None

    def run_on_all_data(self, data_dir: str):
        files = self.find_avro_files(data_dir)
        return self.process_files(files)
