"""
Analyze stress event distribution for participants.
Helps understand why some participants fail in temporal CV.
"""

from pipeline import StressDetectionPipeline
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_DIR = r"D:\workdir\ptsd_stress_detection\refined_tags"

PARTICIPANTS = ["TRAIL009", "TRAIL10"]

def find_tags_file(participant_id: str) -> str:
    pattern = os.path.join(TAGS_DIR, f"{participant_id}_refined_tags*.csv")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getmtime)
    return None

def analyze_stress_distribution():
    print("=== Stress Event Distribution Analysis ===\n")
    
    pipeline = StressDetectionPipeline(use_model="xgboost", n_jobs=-1)
    
    for participant in PARTICIPANTS:
        print(f"\n{'='*60}")
        print(f"Participant: {participant}")
        print('='*60)
        
        tags_file = find_tags_file(participant)
        if not tags_file:
            print("  No tags file found.")
            continue
        
        # Load features
        features = pipeline.process_participant_cached(DATA_DIR, participant, overlap_percent=0.75)
        if features.empty:
            print("  No features.")
            continue
        
        # Load and align labels
        tags = pipeline.load_labels(tags_file)
        labeled = pipeline.align_labels(features, tags)
        
        if labeled.empty:
            print("  No labeled data.")
            continue
        
        # Sort by time
        if 'start_time' in labeled.columns:
            labeled = labeled.sort_values('start_time').reset_index(drop=True)
        
        n_total = len(labeled)
        n_stress = sum(labeled['label'] == 1)
        n_baseline = sum(labeled['label'] == 0)
        
        print(f"Total windows: {n_total}")
        print(f"Stress events: {n_stress} ({100*n_stress/n_total:.1f}%)")
        print(f"Baseline events: {n_baseline}")
        
        # Analyze temporal distribution
        stress_indices = labeled[labeled['label'] == 1].index.tolist()
        
        if len(stress_indices) < 2:
            print("  Not enough stress events to analyze distribution.")
            continue
        
        # Split into 5 temporal segments
        segment_size = n_total // 5
        segment_stress = []
        for i in range(5):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < 4 else n_total
            segment_data = labeled.iloc[start_idx:end_idx]
            stress_in_segment = sum(segment_data['label'] == 1)
            segment_stress.append(stress_in_segment)
            print(f"  Segment {i+1} (idx {start_idx}-{end_idx}): {stress_in_segment} stress events")
        
        # Check for clustering
        max_segment = max(segment_stress)
        segments_with_stress = sum(1 for s in segment_stress if s > 0)
        
        print(f"\nDistribution Analysis:")
        print(f"  Segments with stress: {segments_with_stress}/5")
        print(f"  Max stress in one segment: {max_segment} ({100*max_segment/n_stress:.1f}% of all)")
        
        if segments_with_stress < 3:
            print(f"  ⚠️ WARNING: Stress events clustered in {segments_with_stress} segments!")
            print(f"  This causes TimeSeriesSplit folds to have no stress in test set.")
        
        # Time analysis
        if 'start_time' in labeled.columns:
            stress_times = labeled[labeled['label'] == 1]['start_time']
            baseline_times = labeled[labeled['label'] == 0]['start_time']
            
            # Convert to hours of day
            stress_hours = [(pd.to_datetime(t, unit='s', utc=True).tz_convert('Israel').hour) for t in stress_times]
            
            print(f"\nStress event hours: {sorted(set(stress_hours))}")
            
            # Day span
            stress_days = [(pd.to_datetime(t, unit='s', utc=True).date()) for t in stress_times]
            unique_days = sorted(set(stress_days))
            print(f"Stress events span {len(unique_days)} unique days")
            if len(unique_days) <= 5:
                print(f"  Days: {unique_days}")

if __name__ == "__main__":
    analyze_stress_distribution()
