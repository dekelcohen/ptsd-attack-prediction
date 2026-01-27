"""
Multi-Participant Training with Leave-One-Subject-Out Cross-Validation.

This script:
1. Loads data for multiple participants (with caching)
2. Runs LOSO CV for rigorous validation
3. Optionally tunes hyperparameters
4. Supports focal loss for extreme class imbalance
"""

from pipeline import StressDetectionPipeline
import os
import glob
import pandas as pd

# Configuration
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_DIR = r"D:\workdir\ptsd_stress_detection\refined_tags"

# List of participants with valid refined tags
PARTICIPANTS = [
    "TRAIL009",
    "TRAIL10",
    # Add more participants here as their tags become available
]

# Model settings
USE_FOCAL_LOSS = False  # Focal loss has numerical issues; use scale_pos_weight instead
WINDOW_SIZE_MIN = 10  # 10-minute windows

def find_tags_file(participant_id: str) -> str:
    """Find the refined tags file for a participant."""
    pattern = os.path.join(TAGS_DIR, f"{participant_id}_refined_tags*.csv")
    files = glob.glob(pattern)
    if files:
        # Return most recent
        return max(files, key=os.path.getmtime)
    return None

def run_loso():
    print("=== Multi-Participant LOSO Training ===\n")
    
    # Initialize pipeline with settings
    model_type = "xgboost_focal" if USE_FOCAL_LOSS else "xgboost"
    pipeline = StressDetectionPipeline(
        use_model=model_type, 
        n_jobs=-1,
        window_size_min=WINDOW_SIZE_MIN
    )
    
    all_participant_data = []
    
    for participant in PARTICIPANTS:
        print(f"\n--- Processing {participant} ---")
        
        tags_file = find_tags_file(participant)
        if not tags_file:
            print(f"  No tags file found for {participant}, skipping.")
            continue
        
        # Load features (with caching)
        features = pipeline.process_participant_cached(DATA_DIR, participant, overlap_percent=0.75)
        
        if features.empty:
            print(f"  No features extracted for {participant}, skipping.")
            continue
        
        # Load and align labels
        tags = pipeline.load_labels(tags_file)
        labeled = pipeline.align_labels(features, tags)
        
        if labeled.empty:
            print(f"  No labeled data for {participant}, skipping.")
            continue
        
        # Add participant ID
        labeled['participant_id'] = participant
        
        n_stress = sum(labeled['label'] == 1)
        n_baseline = sum(labeled['label'] == 0)
        print(f"  Loaded {len(labeled)} windows: {n_stress} stress, {n_baseline} baseline")
        
        all_participant_data.append(labeled)
    
    if len(all_participant_data) < 2:
        print("\nNeed at least 2 participants for LOSO CV.")
        print("Falling back to single-participant stratified validation...")
        
        if len(all_participant_data) == 1:
            data = all_participant_data[0]
            X = data.drop(columns=['label', 'start_time', 'end_time', 'source_file', 
                                   'timestamp', 'participant_id'], errors='ignore')
            y = data['label']
            timestamps = data['start_time'] if 'start_time' in data.columns else None
            
            # Run stratified group validation
            if timestamps is not None:
                pipeline.classifier.train_and_evaluate_cv_stratified_groups(X, y, timestamps)
        return
    
    # Combine all data
    combined = pd.concat(all_participant_data, ignore_index=True)
    
    print(f"\n=== Combined Dataset ===")
    print(f"Total windows: {len(combined)}")
    print(f"Participants: {combined['participant_id'].nunique()}")
    print(f"Class balance: {combined['label'].value_counts().to_dict()}")
    print(f"Model: {model_type}, Window: {WINDOW_SIZE_MIN} min")
    
    # Prepare for training
    X = combined.drop(columns=['label', 'start_time', 'end_time', 'source_file', 
                               'timestamp', 'participant_id'], errors='ignore')
    y = combined['label']
    participant_ids = combined['participant_id']
    
    # Run LOSO CV
    mean_f1, mean_prec, mean_rec, results = pipeline.classifier.train_and_evaluate_loso(
        X, y, participant_ids
    )
    
    print("\n=== Per-Participant Results ===")
    for pid, res in results.items():
        print(f"  {pid}: F1={res['f1']:.4f}, P={res['prec']:.2f}, R={res['rec']:.2f} ({res['n_stress']} stress)")

def run_with_tuning():
    """Run with hyperparameter tuning first."""
    print("=== Multi-Participant Training with Hyperparameter Tuning ===\n")
    
    model_type = "xgboost_focal" if USE_FOCAL_LOSS else "xgboost"
    pipeline = StressDetectionPipeline(
        use_model=model_type,
        n_jobs=-1,
        window_size_min=WINDOW_SIZE_MIN
    )
    
    all_participant_data = []
    
    for participant in PARTICIPANTS:
        print(f"\n--- Processing {participant} ---")
        
        tags_file = find_tags_file(participant)
        if not tags_file:
            continue
        
        features = pipeline.process_participant_cached(DATA_DIR, participant, overlap_percent=0.75)
        
        if features.empty:
            continue
        
        tags = pipeline.load_labels(tags_file)
        labeled = pipeline.align_labels(features, tags)
        
        if labeled.empty:
            continue
        
        labeled['participant_id'] = participant
        all_participant_data.append(labeled)
    
    if not all_participant_data:
        print("No data loaded.")
        return
    
    combined = pd.concat(all_participant_data, ignore_index=True)
    
    X = combined.drop(columns=['label', 'start_time', 'end_time', 'source_file', 
                               'timestamp', 'participant_id'], errors='ignore')
    y = combined['label']
    
    # Tune hyperparameters first
    print("\n=== Phase 1: Hyperparameter Tuning ===")
    best_params, best_score = pipeline.classifier.tune_hyperparameters(X, y, n_iter=15)
    
    # Then run LOSO with tuned model
    print("\n=== Phase 2: LOSO Validation with Tuned Model ===")
    participant_ids = combined['participant_id']
    
    if len(participant_ids.unique()) >= 2:
        pipeline.classifier.train_and_evaluate_loso(X, y, participant_ids)
    else:
        timestamps = combined['start_time'] if 'start_time' in combined.columns else None
        if timestamps is not None:
            pipeline.classifier.train_and_evaluate_cv_stratified_groups(X, y, timestamps)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--tune":
        run_with_tuning()
    else:
        run_loso()
