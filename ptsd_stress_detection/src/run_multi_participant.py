from pipeline import StressDetectionPipeline
import os
import glob
import pandas as pd
from imblearn.over_sampling import SMOTE

# Multi-participant Training Script
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags\auto_modified_tags"

# Participants to combine for training
def discover_participants(tags_dir):
    tag_files = glob.glob(os.path.join(tags_dir, "*_valid_tags_modified.csv"))
    return [os.path.basename(f).split("_")[0] for f in tag_files]

PARTICIPANTS = discover_participants(TAGS_DIR)

def run_multi_participant():
    print("=== Running Multi-Participant Training ===")
    pipeline = StressDetectionPipeline(use_model="xgboost")
    
    # Load invalid timestamps from aggregated CSV files (minutes with missing_value_reason)
    pipeline.load_invalid_timestamps_from_aggregated(DATA_DIR)
    
    files = pipeline.find_avro_files(DATA_DIR)
    
    all_participant_data = []
    
    for participant in PARTICIPANTS:
        print(f"\n--- Processing {participant} ---")
        
        # Find participant files
        participant_files = [f for f in files if participant in f]
        print(f"Found {len(participant_files)} files for {participant}")
        
        if len(participant_files) == 0:
            print(f"WARNING: No files found for {participant}")
            continue
            
        # Process files
        participant_features = []
        for i, f in enumerate(participant_files):
            if i % 50 == 0: print(f"[{i}/{len(participant_files)}] Processing...")
            try:
                res = pipeline.process_avro_file(f)
                if res:
                    df_eda, df_hr, df_temp, df_acc = res
                    feats = pipeline.create_features_from_streams(df_eda, df_hr, df_temp, df_acc, overlap_percent=0.5)
                    if not feats.empty:
                        feats['source_file'] = os.path.basename(f)
                        feats['participant'] = participant
                        participant_features.append(feats)
            except Exception as e:
                pass  # Skip failed files silently
        
        if participant_features:
            participant_df = pd.concat(participant_features, ignore_index=True)
            
            # Load and align labels for this participant
            tags_file = os.path.join(TAGS_DIR, f"{participant}_valid_tags_modified.csv")
            if os.path.exists(tags_file):
                tags = pipeline.load_labels(tags_file)
                labeled = pipeline.align_labels(participant_df, tags)
                print(f"{participant}: {len(labeled)} windows, {labeled['label'].sum()} stress events")
                all_participant_data.append(labeled)
            else:
                print(f"WARNING: No tags file found at {tags_file}")
    
    if not all_participant_data:
        print("No data collected from any participant!")
        return
    
    # Combine all participant data
    combined_data = pd.concat(all_participant_data, ignore_index=True)
    
    print(f"\n=== Combined Dataset ===")
    print(f"Total Windows: {len(combined_data)}")
    print(f"Class Balance: {combined_data['label'].value_counts().to_dict()}")
    print(f"Participants: {combined_data['participant'].unique()}")
    
    if len(combined_data['label'].unique()) > 1:
        X = combined_data.drop(columns=['label', 'start_time', 'end_time', 'source_file', 'timestamp', 'participant'], errors='ignore')
        y = combined_data['label']
        print(f"\nFeatures: {list(X.columns)}")
        print(f"Feature count: {len(X.columns)}")
        # Report additional metrics for all models
        for model_type in ["xgboost", "randomforest", "extratrees", "lightgbm"]:
            print(f"\n=== Model: {model_type} ===")
            pipeline.classifier = StressClassifier(model_type=model_type)
            pipeline.classifier.train_and_evaluate_cv_threshold(X, y)
            # Report ROC-AUC and confusion matrix
            from sklearn.metrics import roc_auc_score, confusion_matrix
            y_pred = pipeline.classifier.model.predict(X)
            y_proba = pipeline.classifier.model.predict_proba(X)[:, 1] if hasattr(pipeline.classifier.model, 'predict_proba') else None
            if y_proba is not None:
                auc = roc_auc_score(y, y_proba)
                print(f"ROC-AUC: {auc:.4f}")
            cm = confusion_matrix(y, y_pred)
            print(f"Confusion Matrix:\n{cm}")
    else:
        print("Dataset has only one class. Cannot train.")

if __name__ == "__main__":
    run_multi_participant()
