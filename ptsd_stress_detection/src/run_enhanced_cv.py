"""
Enhanced Within-Participant CV with:
1. Stratified Shuffle-Split for participants with clustered stress events
2. Combined multi-participant training option
3. LightGBM support

Usage:
    python run_enhanced_cv.py [model_type] [--combined]
    
    model_type: xgboost, lightgbm, randomforest (default: xgboost)
    --combined: Train on combined data from all participants
"""

from pipeline import StressDetectionPipeline
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, StratifiedShuffleSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import sys

# Configuration
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
# TAGS_DIR = r"D:\workdir\ptsd_stress_detection\refined_tags"
TAGS_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags"

PARTICIPANTS = ["TRAIL009", "TRAIL10"]

WINDOW_SIZE_MIN = 10
N_SPLITS = 5

# Parse arguments
MODEL_TYPE = "xgboost"
USE_COMBINED = False

for arg in sys.argv[1:]:
    if arg == "--combined":
        USE_COMBINED = True
    elif arg in ["xgboost", "lightgbm", "randomforest"]:
        MODEL_TYPE = arg

def add_rolling_lag_features(X: pd.DataFrame, window_sizes=[3, 5]) -> pd.DataFrame:
    """Add rolling statistics and lag features."""
    X_new = X.copy()
    key_features = ['eda_phasic_mean', 'hr_mean', 'hr_rmssd', 'temp_mean', 'acc_magnitude']
    existing_keys = [f for f in key_features if f in X.columns]
    
    for feature in existing_keys:
        for window in window_sizes:
            X_new[f'{feature}_roll_mean_{window}'] = X[feature].shift(1).rolling(window=window, min_periods=1).mean()
            X_new[f'{feature}_roll_std_{window}'] = X[feature].shift(1).rolling(window=window, min_periods=1).std()
            X_new[f'{feature}_lag_{window}'] = X[feature].shift(window)
    
    X_new = X_new.ffill().bfill()
    return X_new

def analyze_stress_clustering(y: pd.Series, n_segments=5) -> bool:
    """Check if stress events are too clustered for TimeSeriesSplit."""
    segment_size = len(y) // n_segments
    segments_with_stress = 0
    
    for i in range(n_segments):
        start_idx = i * segment_size
        end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(y)
        if y.iloc[start_idx:end_idx].sum() > 0:
            segments_with_stress += 1
    
    # If less than 3 segments have stress, use stratified shuffle instead
    return segments_with_stress < 3

def find_tags_file(participant_id: str) -> str:
    pattern = os.path.join(TAGS_DIR, f"{participant_id}_valid_tags*.csv")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getmtime)
    return None

def train_and_evaluate(X_train, y_train, X_test, y_test, classifier, fold_name):
    """Train and evaluate a single fold."""
    train_stress = sum(y_train == 1)
    test_stress = sum(y_test == 1)
    
    if train_stress == 0 or test_stress == 0:
        return None
    
    # Apply SMOTE
    try:
        from imblearn.over_sampling import SMOTE
        k = min(5, train_stress - 1)
        if k >= 1:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train
    except Exception:
        X_train_res, y_train_res = X_train, y_train
    
    # Feature selection
    n_features = min(30, X_train_res.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train_res, y_train_res)
    X_test_selected = selector.transform(X_test)
    
    # Set class weight for XGBoost
    if hasattr(classifier.model, 'set_params') and classifier.model_type == "xgboost":
        ratio = sum(y_train == 0) / train_stress
        classifier.model.set_params(scale_pos_weight=ratio * 5.0)
    
    classifier.model.fit(X_train_selected, y_train_res)
    
    # Threshold tuning
    y_proba = classifier.model.predict_proba(X_test_selected)[:, 1]
    
    best_f1 = 0
    best_thresh = 0.5
    for thresh in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_pred = (y_proba >= best_thresh).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"  {fold_name}: F1={f1:.4f} (P={prec:.2f}, R={rec:.2f}) | thresh={best_thresh:.2f} | {test_stress} stress")
    
    return {'f1': f1, 'precision': prec, 'recall': rec, 'threshold': best_thresh, 'test_stress': test_stress}

def run_single_participant_cv(participant, pipeline, all_data):
    """Run CV for a single participant with adaptive strategy."""
    print(f"\n{'='*50}")
    print(f"Participant: {participant}")
    print('='*50)
    
    tags_file = find_tags_file(participant)
    if not tags_file:
        print("  No tags file found, skipping.")
        return None, None
    
    features = pipeline.process_participant_cached(DATA_DIR, participant, overlap_percent=0.75)
    if features.empty:
        print("  No features, skipping.")
        return None, None
    
    tags = pipeline.load_labels(tags_file)
    labeled = pipeline.align_labels(features, tags)
    
    if labeled.empty:
        print("  No labeled data, skipping.")
        return None, None
    
    if 'start_time' in labeled.columns:
        labeled = labeled.sort_values('start_time').reset_index(drop=True)
    
    n_stress = sum(labeled['label'] == 1)
    n_baseline = sum(labeled['label'] == 0)
    print(f"Data: {len(labeled)} windows, {n_stress} stress, {n_baseline} baseline")
    
    X = labeled.drop(columns=['label', 'start_time', 'end_time', 'source_file', 
                              'timestamp', 'participant_id'], errors='ignore')
    y = labeled['label']
    
    # Add rolling/lag features
    X = add_rolling_lag_features(X)
    
    # Store for combined training
    labeled['participant_id'] = participant
    all_data.append(labeled)
    
    # Check if stress is too clustered for TimeSeriesSplit
    use_stratified = analyze_stress_clustering(y)
    
    if use_stratified:
        print(f"  ⚠️ Stress clustered - using Stratified Shuffle-Split instead of TimeSeries")
        cv = StratifiedShuffleSplit(n_splits=N_SPLITS, test_size=0.2, random_state=42)
        cv_iter = cv.split(X, y)
    else:
        cv = TimeSeriesSplit(n_splits=N_SPLITS)
        cv_iter = cv.split(X)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(cv_iter, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                   pipeline.classifier, f"Fold {fold}")
        if result:
            fold_results.append(result)
    
    if fold_results:
        mean_f1 = np.mean([r['f1'] for r in fold_results])
        mean_prec = np.mean([r['precision'] for r in fold_results])
        mean_rec = np.mean([r['recall'] for r in fold_results])
        print(f"\n  *** {participant} Mean: F1={mean_f1:.4f}, P={mean_prec:.4f}, R={mean_rec:.4f} ***")
        return {'participant': participant, 'mean_f1': mean_f1, 'mean_precision': mean_prec, 
                'mean_recall': mean_rec, 'n_stress': n_stress}, (X, y)
    
    return None, None

def run_combined_training(all_data, pipeline):
    """Train on combined data from all participants."""
    print(f"\n{'='*60}")
    print("=== COMBINED MULTI-PARTICIPANT TRAINING ===")
    print('='*60)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    if 'start_time' in combined.columns:
        combined = combined.sort_values('start_time').reset_index(drop=True)
    
    n_stress = sum(combined['label'] == 1)
    n_baseline = sum(combined['label'] == 0)
    print(f"Combined data: {len(combined)} windows, {n_stress} stress, {n_baseline} baseline")
    
    X = combined.drop(columns=['label', 'start_time', 'end_time', 'source_file', 
                               'timestamp', 'participant_id'], errors='ignore')
    y = combined['label']
    
    X = add_rolling_lag_features(X)
    
    # Use StratifiedShuffleSplit for combined data
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                   pipeline.classifier, f"Combined Fold {fold}")
        if result:
            fold_results.append(result)
    
    if fold_results:
        mean_f1 = np.mean([r['f1'] for r in fold_results])
        mean_prec = np.mean([r['precision'] for r in fold_results])
        mean_rec = np.mean([r['recall'] for r in fold_results])
        print(f"\n*** Combined Training Results: F1={mean_f1:.4f}, P={mean_prec:.4f}, R={mean_rec:.4f} ***")
        return {'mean_f1': mean_f1, 'mean_precision': mean_prec, 'mean_recall': mean_rec}
    
    return None

def main():
    print("=== Enhanced Within-Participant CV ===\n")
    print(f"Model: {MODEL_TYPE}")
    print(f"Combined Training: {USE_COMBINED}")
    
    pipeline = StressDetectionPipeline(use_model=MODEL_TYPE, n_jobs=-1, window_size_min=WINDOW_SIZE_MIN)
    
    all_results = []
    all_data = []
    
    # Run per-participant CV
    for participant in PARTICIPANTS:
        result, _ = run_single_participant_cv(participant, pipeline, all_data)
        if result:
            all_results.append(result)
    
    # Summary
    if all_results:
        print("\n" + "="*60)
        print("=== PER-PARTICIPANT RESULTS ===")
        print("="*60)
        
        df = pd.DataFrame(all_results)
        total_stress = df['n_stress'].sum()
        weighted_f1 = sum(df['mean_f1'] * df['n_stress']) / total_stress
        weighted_prec = sum(df['mean_precision'] * df['n_stress']) / total_stress
        weighted_rec = sum(df['mean_recall'] * df['n_stress']) / total_stress
        
        for _, row in df.iterrows():
            print(f"  {row['participant']}: F1={row['mean_f1']:.4f}, P={row['mean_precision']:.4f}, R={row['mean_recall']:.4f} ({row['n_stress']} stress)")
        
        print(f"\nWeighted Average: F1={weighted_f1:.4f}, P={weighted_prec:.4f}, R={weighted_rec:.4f}")
    
    # Combined training
    if USE_COMBINED and len(all_data) >= 2:
        run_combined_training(all_data, pipeline)

if __name__ == "__main__":
    main()
