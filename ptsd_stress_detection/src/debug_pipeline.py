from pipeline import StressDetectionPipeline
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Debug Config
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_FILE = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags\auto_modified_tags\TRAIL009_valid_tags_modified.csv"
TARGET_PARTICIPANT = "TRAIL009"

def run_debug():
    print("=== DEBUGGING PIPELINE INTEGRITY ===")
    pipeline = StressDetectionPipeline(use_model="xgboost")
    
    # Load invalid timestamps from aggregated CSV files (minutes with missing_value_reason)
    pipeline.load_invalid_timestamps_from_aggregated(DATA_DIR)
    
    # 1. Find Files
    files = pipeline.find_avro_files(DATA_DIR)
    subset_files = [f for f in files if TARGET_PARTICIPANT in f][:100] # Limit to 100 files for speed
    print(f"Debugging with {len(subset_files)} files...")
    
    # 2. Extract
    all_data = []
    for i, f in enumerate(subset_files):
        if i % 10 == 0: print(f"Processing {i}...")
        try:
            res = pipeline.process_avro_file(f)
            if res:
                df_eda, df_hr, df_temp, df_acc = res
                feats = pipeline.create_features_from_streams(df_eda, df_hr, df_temp, df_acc)
                if not feats.empty:
                    all_data.append(feats)
        except Exception as e:
            print(f"Warning: Failed to process {f}: {e}")
            
    if not all_data:
        print("[CRITICAL FAIL] No data extracted.")
        return

    features = pd.concat(all_data, ignore_index=True)
    
    # 3. INSPECT DATAFRAME
    print("\n--- DATAFRAME INSPECTION ---")
    print(f"Shape: {features.shape}")
    print("Columns:", features.columns.tolist())
    
    # Check for Temp/Acc presence
    new_cols = ['temp_mean', 'acc_mean', 'acc_std']
    for col in new_cols:
        if col in features.columns:
            non_null = features[col].count()
            mean_val = features[col].mean()
            print(f"[OK] {col} present. Non-null: {non_null}/{len(features)}. Mean: {mean_val:.4f}")
        else:
            print(f"[FAIL] {col} MISSING from dataframe!")

    # 4. Feature Importance
    print("\n--- FEATURE IMPORTANCE CHECK ---")
    tags = pipeline.load_labels(TAGS_FILE)
    labeled = pipeline.align_labels(features, tags)
    
    X = labeled.drop(columns=['label', 'start_time', 'end_time', 'source_file', 'timestamp', 'hour_of_day'], errors='ignore')
    y = labeled['label']
    
    if len(y.unique()) > 1:
        pipeline.classifier.train(X, y)
        
        # Get importance
        importances = pipeline.classifier.model.feature_importances_
        feature_names = X.columns
        
        feat_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feat_imp = feat_imp.sort_values('importance', ascending=False)
        
        print("\nTop 10 Features:")
        print(feat_imp.head(10))
        
        # Check if new features are in use
        print("\nNew Features Ranks:")
        print(feat_imp[feat_imp['feature'].isin(new_cols)])
        
    else:
        print("Not enough classes to train debug model.")

if __name__ == "__main__":
    run_debug()
