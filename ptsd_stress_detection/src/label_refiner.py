import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pipeline import StressDetectionPipeline
from datetime import timedelta

class LabelRefiner:
    def __init__(self, data_dir, cache_dir="refinement_cache"):
        self.data_dir = data_dir
        self.pipeline = StressDetectionPipeline(cache_dir=cache_dir)
        self.pipeline.classifier = None # We don't need the classifier, just features
        
        # Lazy loading of invalid timestamps will be done per participant in refine_labels
        pass
        
    def parse_idt_time(self, ts_str):
        # Format: 2025-08-25 19:05:53 IDT
        clean_ts = ts_str.replace(" IDT", "").replace(" IST", "")
        dt = pd.to_datetime(clean_ts)
        return dt.tz_localize('Israel')

    def load_raw_tags(self, tags_file):
        df = pd.read_csv(tags_file)
        # Parse timestamp
        df['dt'] = df['timestamp'].apply(self.parse_idt_time)
        df['unix_timestamp'] = df['dt'].astype('int64') // 10**9
        return df

    def get_participant_features(self, participant_id):
        print(f"Loading all data for {participant_id}...")
        files = self.pipeline.find_avro_files(self.data_dir)
        p_files = [f for f in files if participant_id in f]
        features_df = self.pipeline.process_files(p_files)
        if features_df is None or features_df.empty:
            return pd.DataFrame()
            
        if 'timestamp' not in features_df.columns and 'start_time' in features_df.columns:
            features_df['timestamp'] = features_df['start_time']
            
        features_df = features_df.sort_values('timestamp').reset_index(drop=True)
        return features_df

    def refine_labels(self, participant_id, tags_file, output_file):
        # 1. Load Data
        # Load invalid timestamps specific to this participant (uses caching)
        print(f"Loading invalid timestamps for {participant_id}...")
        self.pipeline.load_invalid_timestamps_from_aggregated(self.data_dir, participant_id=participant_id)
        
        features_df = self.get_participant_features(participant_id)
        if features_df.empty:
            print("No features found!")
            return
            
        # 2. Train Isolation Forest
        print("Training Isolation Forest on all participant data...")
        drop_cols = ['label', 'start_time', 'end_time', 'source_file', 'timestamp', 'hour_of_day']
        all_cols = features_df.columns
        stress_cols = [c for c in all_cols if c not in drop_cols and ('eda' in c.lower() or 'hr' in c.lower())]
        
        print(f"Using {len(stress_cols)} features for anomaly detection: {stress_cols}")
        X = features_df[stress_cols].fillna(0)
        
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Sensitivity 0.15
        iso = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
        features_df['anomaly_score'] = iso.fit_predict(X_scaled) 
        features_df['anomaly_score_cont'] = -iso.decision_function(X_scaled) 
        
        # 3. Load Tags
        tags_df = self.load_raw_tags(tags_file)
        
        refined_tags = []
        print(f"Refining {len(tags_df)} tags...")
        
        for _, tag in tags_df.iterrows():
            tag_time = tag['unix_timestamp']
            
            # Constraint: Max 6h before (-21600s) or 30m after (+1800s)
            start_constraint = tag_time - (6 * 3600)
            end_constraint = tag_time + (30 * 60)
            
            # Visualization Window: -12h to +4h
            viz_start = tag_time - (12 * 3600)
            viz_end = tag_time + (4 * 3600)
            
            # Slice window for visualization AND calculation
            viz_mask = (features_df['timestamp'] >= viz_start) & (features_df['timestamp'] <= viz_end)
            viz_df = features_df[viz_mask].copy()
            
            if viz_df.empty:
                # No data - keep original time but convert to consistent ISO format
                new_dt = pd.to_datetime(tag_time, unit='s', utc=True).tz_convert('Israel')
                tag['new_timestamp'] = new_dt.isoformat()
                refined_tags.append(tag)
                continue
                
            # Filter for Peaks "above threshold"
            # Threshold corresponds to the contamination percentile (top 15%)
            global_thresh = features_df['anomaly_score_cont'].quantile(1.0 - 0.15)
            
            # Find peaks in the viz_df scores
            scores = viz_df['anomaly_score_cont'].values
            timestamps = viz_df['timestamp'].values
            
            peaks, _ = find_peaks(scores, height=global_thresh, distance=5)
            
            refined_time = tag_time 
            
            if len(peaks) > 0:
                peak_times = timestamps[peaks]
                
                # Filter peaks within constraint window [-6h, +30m]
                valid_mask = (peak_times >= start_constraint) & (peak_times <= end_constraint)
                valid_peak_times = peak_times[valid_mask]
                
                if len(valid_peak_times) > 0:
                    # Find closest peak to tag
                    time_diffs = np.abs(valid_peak_times - tag_time)
                    best_idx = np.argmin(time_diffs)
                    refined_time = valid_peak_times[best_idx]
                else:
                    print(f"  [Info] No peaks in valid window [-6h, +30m] for tag {tag['dt']}")

            # Visualization
            plot_dir = os.path.join("refinement_plots", participant_id)
            plot_path = os.path.join(plot_dir, f"tag_{tag['timestamp'].replace(':','-').replace(' ','_')}.png")
            self.plot_window(viz_df, tag['dt'], refined_time, global_thresh, plot_path, viz_start, viz_end)
            
            # Save Refined Time - always in ISO format with Israel timezone
            new_dt = pd.to_datetime(refined_time, unit='s', utc=True).tz_convert('Israel')
            tag['new_timestamp'] = new_dt.isoformat()
            
            refined_tags.append(tag)
            
        # Save Result
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            result_df = pd.DataFrame(refined_tags)
            result_df.to_csv(output_file, index=False)
            print(f"Saved refined tags to {output_file}")
        except PermissionError:
            print(f"ERROR: Could not save CSV to {output_file}. File is open.")

    def plot_window(self, window_df, original_tag_dt, refined_ts, threshold, output_path, xlim_start, xlim_end):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # Plot Anomaly Score
            times = pd.to_datetime(window_df['timestamp'], unit='s', utc=True).dt.tz_convert('Israel')
            scores = window_df['anomaly_score_cont']
            
            ax.plot(times, scores, label='Anomaly Score', color='blue', alpha=0.7)
            ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold (Top 15%)')
            
            # Original Tag
            ax.axvline(original_tag_dt, color='orange', linestyle='-', linewidth=2, label='Original Tag')
            
            # Refined Tag (Peak)
            refined_dt = pd.to_datetime(refined_ts, unit='s', utc=True).tz_convert('Israel')
            ax.axvline(refined_dt, color='green', linestyle='--', linewidth=2, label='Refined Peak')
            
            # Explicit X-Limits
            xlim_start_dt = pd.to_datetime(xlim_start, unit='s', utc=True).tz_convert('Israel')
            xlim_end_dt = pd.to_datetime(xlim_end, unit='s', utc=True).tz_convert('Israel')
            ax.set_xlim(xlim_start_dt, xlim_end_dt)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=xlim_start_dt.tz))
            
            ax.set_title(f"Refinement: {original_tag_dt} -> {refined_dt}\n(Window: -12h to +4h, Peak Detection)")
            ax.set_xlabel("Time (Israel)")
            ax.set_ylabel("Anomaly Score")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close(fig)
        except Exception as e:
            print(f"Plotting failed: {e}")

if __name__ == "__main__":
    DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
    RAW_TAGS_FILE = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags\raw_tags\TRAIL10_valid_tags_raw.csv"
    OUTPUT_FILE = r"D:\workdir\ptsd_stress_detection\refined_tags\TRAIL10_refined_tags_test.csv"
    
    refiner = LabelRefiner(DATA_DIR)
    refiner.refine_labels("TRAIL10", RAW_TAGS_FILE, OUTPUT_FILE)
