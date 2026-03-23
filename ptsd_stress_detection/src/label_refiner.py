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
    """
    Enhanced Label Refiner with origin-aware confidence scoring.
    
    Handles different tag origins:
    - watch: User pressed button on watch (immediate, but may be accidental)
    - remote: User answered questionnaire on phone
    - app: User reported directly from phone app
    
    Assigns confidence scores based on:
    - Origin pattern (watch+remote = highest confidence)
    - Severity rating (-1 = missing, 1-5 = user-rated)
    - Physiological validation (anomaly detection)
    """
    
    def __init__(self, data_dir, cache_dir="refinement_cache"):
        self.data_dir = data_dir
        self.pipeline = StressDetectionPipeline(cache_dir=cache_dir)
        self.pipeline.classifier = None  # We don't need the classifier
        
    def parse_idt_time(self, ts_str):
        """Parse timestamp string with IDT/IST timezone."""
        clean_ts = ts_str.replace(" IDT", "").replace(" IST", "")
        dt = pd.to_datetime(clean_ts)
        return dt.tz_localize('Israel')

    def load_raw_tags(self, tags_file):
        """Load raw tags CSV with origin information."""
        df = pd.read_csv(tags_file)
        df['dt'] = df['timestamp'].apply(self.parse_idt_time)
        df['unix_timestamp'] = df['dt'].astype('int64') // 10**9
        return df

    def get_participant_features(self, participant_id):
        """Load all features for a participant."""
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

    def merge_duplicate_tags(self, tags_df, time_window_min=5):
        """
        Merge tags that are within time_window_min of each other.
        
        Strategy:
        - Group tags within time window
        - Use earliest timestamp
        - Combine origins (watch+remote = highest confidence)
        - Keep highest severity
        """
        if tags_df.empty:
            return tags_df
            
        tags_df = tags_df.sort_values('unix_timestamp').reset_index(drop=True)
        
        merged_tags = []
        used_indices = set()
        window_sec = time_window_min * 60
        
        for i, tag in tags_df.iterrows():
            if i in used_indices:
                continue
                
            # Find all tags within window
            tag_time = tag['unix_timestamp']
            mask = (tags_df['unix_timestamp'] >= tag_time) & \
                   (tags_df['unix_timestamp'] <= tag_time + window_sec)
            group = tags_df[mask]
            
            # Mark all as used
            used_indices.update(group.index.tolist())
            
            # Merge info
            merged = tag.copy()
            merged['merged_count'] = len(group)
            
            # Collect origins
            origins = group['origin'].unique().tolist()
            merged['origins_combined'] = '+'.join(sorted(origins))
            
            # Determine origin type
            if 'watch' in origins and 'remote' in origins:
                merged['origin_type'] = 'watch_remote'
            elif 'watch' in origins:
                merged['origin_type'] = 'watch_only'
            elif 'remote' in origins:
                merged['origin_type'] = 'remote_only'
            else:
                merged['origin_type'] = 'app_only'
            
            # Use highest severity (ignore -1)
            valid_severities = group[group['severity'] >= 0]['severity']
            if len(valid_severities) > 0:
                merged['severity'] = valid_severities.max()
            else:
                merged['severity'] = -1
            
            # Use earliest timestamp
            merged['unix_timestamp'] = group['unix_timestamp'].min()
            merged['dt'] = pd.to_datetime(merged['unix_timestamp'], unit='s', utc=True).tz_convert('Israel')
            
            merged_tags.append(merged)
        
        result = pd.DataFrame(merged_tags)
        print(f"Merged {len(tags_df)} tags into {len(result)} unique events")
        return result

    def calculate_confidence(self, tag):
        """
        Calculate confidence score (0-1) for a tag based on origin and severity.
        
        Confidence factors:
        - Origin: watch+remote (1.0), watch_only (0.7), remote_only (0.6), app_only (0.5)
        - Severity: -1 (0.5 multiplier), 1-2 (0.7), 3-4 (0.9), 5 (1.0)
        """
        # Base confidence from origin
        origin_type = tag.get('origin_type', tag.get('origin', 'unknown'))
        
        origin_scores = {
            'watch_remote': 1.0,
            'watch_only': 0.7,
            'watch': 0.7,
            'remote_only': 0.6,
            'remote': 0.6,
            'app_only': 0.5,
            'app': 0.5,
            'unknown': 0.4
        }
        base_score = origin_scores.get(origin_type, 0.4)
        
        # Severity modifier
        severity = tag.get('severity', -1)
        if severity == -1:
            severity_mod = 0.5
        elif severity <= 2:
            severity_mod = 0.7
        elif severity <= 4:
            severity_mod = 0.9
        else:
            severity_mod = 1.0
        
        confidence = base_score * severity_mod
        return round(confidence, 2)

    def get_search_window(self, confidence):
        """
        Get adaptive search window based on confidence.
        
        High confidence: User was real-time, search narrow window
        Low confidence: User may have delayed, search wider window
        
        Returns (backward_seconds, forward_seconds)
        """
        if confidence >= 0.8:
            return (30 * 60, 15 * 60)  # ±30min back, +15min forward
        elif confidence >= 0.5:
            return (2 * 3600, 30 * 60)  # -2h, +30min
        else:
            return (6 * 3600, 30 * 60)  # -6h, +30min

    def walk_back_cluster(self, anchor_time, all_peak_times, cluster_gap_sec=1800):
        """
        Walk backward from anchor peak to find the start of the anomaly cluster.
        
        This finds the true onset of stress by following connected anomaly peaks
        that are within cluster_gap_sec of each other.
        
        From tags_modifing.py: This is critical for finding when stress actually
        started, not just the peak the user noticed.
        """
        if len(all_peak_times) == 0:
            return anchor_time
        
        # Sort all peaks and find ones before anchor
        sorted_peaks = np.sort(all_peak_times)
        precursors = sorted_peaks[sorted_peaks <= anchor_time]
        
        if len(precursors) == 0:
            return anchor_time
        
        # Start from anchor and walk backward
        cluster_start = anchor_time
        
        # Iterate backward through precursors
        for i in range(len(precursors) - 2, -1, -1):  # -2 because last is anchor
            prev_peak = precursors[i]
            gap = cluster_start - prev_peak
            
            if gap <= cluster_gap_sec:
                # Connected to cluster, move start earlier
                cluster_start = prev_peak
            else:
                # Gap too large, cluster broken
                break
        
        return cluster_start

    def validate_with_physiology(self, tag_time, features_df, threshold, window_sec=1800):
        """
        Check if there's a physiological anomaly near the tag time.
        
        Returns (validated: bool, peak_time: float or None, peak_score: float)
        """
        start = tag_time - window_sec
        end = tag_time + window_sec
        
        mask = (features_df['timestamp'] >= start) & (features_df['timestamp'] <= end)
        window = features_df[mask]
        
        if window.empty:
            return False, None, 0.0
        
        # Check if any score exceeds threshold
        max_score = window['anomaly_score_cont'].max()
        if max_score >= threshold:
            peak_idx = window['anomaly_score_cont'].idxmax()
            peak_time = window.loc[peak_idx, 'timestamp']
            return True, peak_time, max_score
        
        return False, None, max_score

    def refine_labels(self, participant_id, tags_file, output_file, min_confidence=0.2):
        """
        Main refinement pipeline with origin-aware processing.
        
        Steps:
        1. Load and merge duplicate tags
        2. Calculate confidence scores
        3. Train Isolation Forest for anomaly detection
        4. Refine each tag with adaptive windows
        5. Validate with physiology
        6. Save results with confidence scores
        """
        # 1. Load invalid timestamps
        print(f"Loading invalid timestamps for {participant_id}...")
        self.pipeline.load_invalid_timestamps_from_aggregated(self.data_dir, participant_id=participant_id)
        
        # 2. Load features
        features_df = self.get_participant_features(participant_id)
        if features_df.empty:
            print("No features found!")
            return
        
        # 3. Load and merge tags
        print("Loading and merging tags...")
        tags_df = self.load_raw_tags(tags_file)
        tags_df = self.merge_duplicate_tags(tags_df, time_window_min=5)
        
        # 4. Calculate confidence scores
        print("Calculating confidence scores...")
        tags_df['confidence'] = tags_df.apply(self.calculate_confidence, axis=1)
        # Lower minimum confidence threshold for tag inclusion
        min_confidence = 0.1
        
        # Show confidence distribution
        print(f"\nConfidence Distribution:")
        for origin_type in tags_df['origin_type'].unique():
            subset = tags_df[tags_df['origin_type'] == origin_type]
            print(f"  {origin_type}: n={len(subset)}, avg_conf={subset['confidence'].mean():.2f}")
        
        # 5. Train Isolation Forest
        print("\nTraining Isolation Forest on all participant data...")
        drop_cols = ['label', 'start_time', 'end_time', 'source_file', 'timestamp', 
                     'hour_of_day', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 
                     'day_sin', 'day_cos']  # Exclude time features from anomaly detection
        all_cols = features_df.columns
        # Use ALL physiological features (same as main model), not just EDA/HR
        # This includes: EDA, HR, temperature, accelerometer, cross-modal features
        stress_cols = [c for c in all_cols if c not in drop_cols and 
                      any(prefix in c.lower() for prefix in ['eda', 'hr', 'temp', 'acc', 'cross'])]
        # Add multi-modal anomaly detection: combine Isolation Forest with rolling z-score
        X = features_df[stress_cols].fillna(0)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        iso = IsolationForest(contamination=0.15, random_state=42, n_jobs=-1)
        features_df['anomaly_score'] = iso.fit_predict(X_scaled) 
        features_df['anomaly_score_cont'] = -iso.decision_function(X_scaled)
        # Rolling z-score anomaly (compute per-feature z and aggregate row-wise max)
        rolling_mean = features_df[stress_cols].rolling(30, min_periods=1).mean()
        rolling_std = features_df[stress_cols].rolling(30, min_periods=1).std().fillna(0)
        rolling_zscores = (features_df[stress_cols] - rolling_mean) / (rolling_std + 1e-6)
        features_df['rolling_zscore_max'] = rolling_zscores.max(axis=1)
        # Combine anomaly scores
        features_df['combined_anomaly'] = features_df['anomaly_score_cont'] + features_df['rolling_zscore_max']
        global_thresh = features_df['combined_anomaly'].quantile(0.85)
        
        # 6. Refine each tag
        refined_tags = []
        print(f"\nRefining {len(tags_df)} tags (min_confidence={min_confidence})...")
        
        for _, tag in tags_df.iterrows():
            confidence = tag['confidence']
            severity = tag.get('severity', -1)
            
            # Skip low confidence tags
            if confidence < min_confidence:
                print(f"  [Skip] Tag {tag['dt']} - confidence {confidence:.2f} below threshold")
                continue
            
            # Skip tags with severity < 1 (from tags_modifing.py logic)
            if severity >= 0 and severity < 1:
                print(f"  [Skip] Tag {tag['dt']} - severity {severity} too low")
                continue
            
            tag_time = tag['unix_timestamp']
            
            # Adaptive search window
            back_sec, fwd_sec = self.get_search_window(confidence)
            start_constraint = tag_time - back_sec
            end_constraint = tag_time + fwd_sec
            
            # Viz window (larger for plotting)
            viz_start = tag_time - (12 * 3600)
            viz_end = tag_time + (4 * 3600)
            
            viz_mask = (features_df['timestamp'] >= viz_start) & (features_df['timestamp'] <= viz_end)
            viz_df = features_df[viz_mask].copy()
            
            if viz_df.empty:
                new_dt = pd.to_datetime(tag_time, unit='s', utc=True).tz_convert('Israel')
                tag['new_timestamp'] = new_dt.isoformat()
                tag['validated'] = False
                refined_tags.append(tag)
                continue
            
            # Find peaks
            scores = viz_df['anomaly_score_cont'].values
            timestamps = viz_df['timestamp'].values
            peaks, _ = find_peaks(scores, height=global_thresh, distance=5)
            
            refined_time = tag_time
            validated = False
            
            if len(peaks) > 0:
                peak_times = timestamps[peaks]
                
                # Filter peaks within constraint window
                valid_mask = (peak_times >= start_constraint) & (peak_times <= end_constraint)
                valid_peak_times = peak_times[valid_mask]
                
                if len(valid_peak_times) > 0:
                    # Find closest peak to tag
                    time_diffs = np.abs(valid_peak_times - tag_time)
                    best_idx = np.argmin(time_diffs)
                    closest_peak = valid_peak_times[best_idx]
                    
                    # Walk back through cluster to find true onset (from tags_modifing.py)
                    refined_time = self.walk_back_cluster(
                        anchor_time=closest_peak,
                        all_peak_times=peak_times,  # Use all peaks for cluster detection
                        cluster_gap_sec=30 * 60  # 30 min gap (matching tags_modifing.py)
                    )
                    validated = True
                else:
                    info_msg = f"No peaks in window [{-back_sec//60}min, +{fwd_sec//60}min]"
                    print(f"  [Info] {tag['dt']}: {info_msg}")
            
            # Additional validation
            if not validated:
                validated, _, _ = self.validate_with_physiology(tag_time, features_df, global_thresh)
            
            # Plot
            plot_dir = os.path.join("refinement_plots", participant_id)
            plot_path = os.path.join(plot_dir, f"tag_{tag['timestamp'].replace(':','-').replace(' ','_')}.png")
            self.plot_window(viz_df, tag['dt'], refined_time, global_thresh, plot_path, 
                           viz_start, viz_end, confidence, tag.get('origin_type', 'unknown'))
            
            # Save refined time
            new_dt = pd.to_datetime(refined_time, unit='s', utc=True).tz_convert('Israel')
            tag['new_timestamp'] = new_dt.isoformat()
            tag['validated'] = validated
            
            refined_tags.append(tag)
        
        # 7. Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        try:
            result_df = pd.DataFrame(refined_tags)
            
            # Reorder columns for readability
            priority_cols = ['timestamp', 'new_timestamp', 'confidence', 'origin_type', 
                           'validated', 'severity', 'merged_count']
            other_cols = [c for c in result_df.columns if c not in priority_cols]
            result_df = result_df[[c for c in priority_cols if c in result_df.columns] + other_cols]
            
            # Origin-priority deduplication (from tags_modifing.py)
            # If multiple tags refined to same timestamp, keep highest priority origin
            if 'origin' in result_df.columns or 'origin_type' in result_df.columns:
                origin_col = 'origin_type' if 'origin_type' in result_df.columns else 'origin'
                
                # Map origin to priority (lower is better): app > remote > watch
                origin_priority = {
                    'app': 1, 'app_only': 1,
                    'remote': 2, 'remote_only': 2,
                    'watch': 3, 'watch_only': 3,
                    'watch_remote': 0  # Best: has both watch and remote confirmation
                }
                result_df['_origin_priority'] = result_df[origin_col].str.lower().map(origin_priority).fillna(4)
                
                # Sort by new_timestamp and priority, keep first (best) for each timestamp
                before_dedup = len(result_df)
                result_df = result_df.sort_values(by=['new_timestamp', '_origin_priority'], ascending=[True, True])
                result_df = result_df.drop_duplicates(subset=['new_timestamp'], keep='first')
                result_df = result_df.drop(columns=['_origin_priority'])
                
                if before_dedup > len(result_df):
                    print(f"  Deduplicated: {before_dedup} -> {len(result_df)} tags (same new_timestamp)")
            
            result_df.to_csv(output_file, index=False)
            print(f"\nSaved {len(result_df)} refined tags to {output_file}")
            
            # Summary
            print(f"\nRefinement Summary:")
            print(f"  Original tags: {len(tags_df)}")
            print(f"  After filtering: {len(result_df)}")
            print(f"  Validated (physio match): {sum(result_df['validated'])}")
            print(f"  Avg confidence: {result_df['confidence'].mean():.2f}")
            
        except PermissionError:
            print(f"ERROR: Could not save CSV to {output_file}. File is open.")

    def plot_window(self, window_df, original_tag_dt, refined_ts, threshold, output_path, 
                   xlim_start, xlim_end, confidence=None, origin_type=None):
        """Plot refinement visualization with confidence info."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(15, 6))
            
            times = pd.to_datetime(window_df['timestamp'], unit='s', utc=True).dt.tz_convert('Israel')
            scores = window_df['anomaly_score_cont']
            
            ax.plot(times, scores, label='Anomaly Score', color='blue', alpha=0.7)
            ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold (Top 15%)')
            ax.axvline(original_tag_dt, color='orange', linestyle='-', linewidth=2, label='Original Tag')
            
            refined_dt = pd.to_datetime(refined_ts, unit='s', utc=True).tz_convert('Israel')
            ax.axvline(refined_dt, color='green', linestyle='--', linewidth=2, label='Refined Peak')
            
            xlim_start_dt = pd.to_datetime(xlim_start, unit='s', utc=True).tz_convert('Israel')
            xlim_end_dt = pd.to_datetime(xlim_end, unit='s', utc=True).tz_convert('Israel')
            ax.set_xlim(xlim_start_dt, xlim_end_dt)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=xlim_start_dt.tz))
            
            # Enhanced title with confidence
            conf_str = f", Conf={confidence:.2f}" if confidence else ""
            origin_str = f", Origin={origin_type}" if origin_type else ""
            ax.set_title(f"Refinement: {original_tag_dt} -> {refined_dt}{conf_str}{origin_str}")
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
    TAGS_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags\raw_tags"
    PERIODS_FILE = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\participant_data_periods.xlsx"
    OUTPUT_DIR = r"D:\workdir\ptsd-attack-prediction\ptsd_stress_detection\refined_tags"

    def load_participants_from_periods(periods_file: str):
        if not os.path.exists(periods_file):
            raise FileNotFoundError(f"participant periods file not found: {periods_file}")

        periods_df = pd.read_excel(periods_file)
        if 'user_id' not in periods_df.columns:
            raise ValueError(f"Expected 'user_id' column in {periods_file}")

        participants = sorted({
            str(pid).strip() for pid in periods_df['user_id'].dropna().astype(str)
            if str(pid).strip().upper().startswith("TRAIL")
        })

        participants = [pid for pid in participants if pid.upper() != "TRAIL007"]
        return participants
    
    participants = load_participants_from_periods(PERIODS_FILE)
    print(f"Loaded {len(participants)} participants from periods file (excluding TRAIL007)")
    
    refiner = LabelRefiner(DATA_DIR)
    
    for participant_id in participants:
        raw_tags_file = os.path.join(TAGS_DIR, f"{participant_id}_valid_tags_raw.csv")
        output_file = os.path.join(OUTPUT_DIR, f"{participant_id}_refined_tags_v2.csv")
        
        # Skip if output already exists
        if os.path.exists(output_file):
            print(f"Skipping {participant_id}: Output file already exists at {output_file}")
            continue
            
        if os.path.exists(raw_tags_file):
            print(f"\n{'='*60}")
            print(f"Processing {participant_id}")
            print('='*60)
            try:
                refiner.refine_labels(participant_id, raw_tags_file, output_file, min_confidence=0.2)
            except Exception as e:
                print(f"Error processing {participant_id}: {e}")
        else:
            print(f"Tags file not found for {participant_id}: {raw_tags_file}")
