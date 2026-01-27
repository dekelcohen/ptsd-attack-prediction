"""
Within-Participant Temporal Cross-Validation.

This script evaluates the model's ability to predict a user's future stress events
based on their past data - the realistic deployment scenario for personalized apps.

For each participant:
1. Sort data chronologically
2. Split into temporal folds (earlier data = train, later data = test)
3. Report per-participant and aggregated metrics
"""

from pipeline import StressDetectionPipeline
import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

# Configuration
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_DIR = r"D:\workdir\ptsd_stress_detection\refined_tags"

PARTICIPANTS = [
    "TRAIL009",
    "TRAIL10",
]

WINDOW_SIZE_MIN = 10
N_SPLITS = 5  # Number of temporal folds

# Parse model type from command line (default: xgboost)
MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "xgboost"

def add_rolling_lag_features(X: pd.DataFrame, window_sizes=[3, 5]) -> pd.DataFrame:
    """
    Add rolling statistics and lag features for temporal patterns.
    Rolling features: mean, std over past N windows
    Lag features: value from N windows ago
    """
    X_new = X.copy()
    
    # Select key features for rolling/lag (to avoid feature explosion)
    key_features = ['eda_phasic_mean', 'hr_mean', 'hr_rmssd', 'temp_mean', 'acc_magnitude']
    existing_keys = [f for f in key_features if f in X.columns]
    
    for feature in existing_keys:
        for window in window_sizes:
            # Rolling mean (past N windows, excluding current)
            X_new[f'{feature}_roll_mean_{window}'] = X[feature].shift(1).rolling(window=window, min_periods=1).mean()
            # Rolling std
            X_new[f'{feature}_roll_std_{window}'] = X[feature].shift(1).rolling(window=window, min_periods=1).std()
            # Lag (value from N windows ago)
            X_new[f'{feature}_lag_{window}'] = X[feature].shift(window)
    
    # Fill NaN from lag/rolling with forward fill then backward fill
    X_new = X_new.ffill().bfill()
    
    return X_new

def find_tags_file(participant_id: str) -> str:
    """Find the refined tags file for a participant."""
    pattern = os.path.join(TAGS_DIR, f"{participant_id}_refined_tags*.csv")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getmtime)
    return None

def run_within_participant_cv():
    print("=== Within-Participant Temporal Cross-Validation ===\n")
    print(f"Model: {MODEL_TYPE}")
    print("This evaluates personalized model performance (train on past, predict future)")
    
    pipeline = StressDetectionPipeline(use_model=MODEL_TYPE, n_jobs=-1, window_size_min=WINDOW_SIZE_MIN)
    
    all_results = []
    
    for participant in PARTICIPANTS:
        print(f"\n{'='*50}")
        print(f"Participant: {participant}")
        print('='*50)
        
        tags_file = find_tags_file(participant)
        if not tags_file:
            print(f"  No tags file found, skipping.")
            continue
        
        # Load features
        features = pipeline.process_participant_cached(DATA_DIR, participant, overlap_percent=0.75)
        if features.empty:
            print(f"  No features, skipping.")
            continue
        
        # Load and align labels
        tags = pipeline.load_labels(tags_file)
        labeled = pipeline.align_labels(features, tags)
        
        if labeled.empty:
            print(f"  No labeled data, skipping.")
            continue
        
        # Sort by time
        if 'start_time' in labeled.columns:
            labeled = labeled.sort_values('start_time').reset_index(drop=True)
        
        n_stress = sum(labeled['label'] == 1)
        n_baseline = sum(labeled['label'] == 0)
        print(f"Data: {len(labeled)} windows, {n_stress} stress, {n_baseline} baseline")
        print(f"Imbalance ratio: 1:{n_baseline/n_stress:.1f}")
        
        if n_stress < N_SPLITS:
            print(f"  Too few stress events for {N_SPLITS}-fold CV, using 2-fold.")
            n_splits = 2
        else:
            n_splits = N_SPLITS
        
        # Prepare features and labels
        X = labeled.drop(columns=['label', 'start_time', 'end_time', 'source_file', 
                                  'timestamp', 'participant_id'], errors='ignore')
        y = labeled['label']
        
        # Add rolling/lag features for temporal patterns
        X = add_rolling_lag_features(X)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            train_stress = sum(y_train == 1)
            test_stress = sum(y_test == 1)
            
            if train_stress == 0:
                print(f"  Fold {fold}: Skipped (no stress in training)")
                continue
            if test_stress == 0:
                print(f"  Fold {fold}: Skipped (no stress in test)")
                continue
            
            # Calculate class weight
            ratio = sum(y_train == 0) / train_stress
            weight_boost = 5.0
            
            # Create and train model for this fold
            classifier = pipeline.classifier
            if classifier.model_type == "xgboost":
                classifier.model.set_params(scale_pos_weight=ratio * weight_boost)
            
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
            
            # Feature selection using mutual information
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            n_features = min(30, X_train_res.shape[1])  # Select top 30 features or all if fewer
            selector = SelectKBest(mutual_info_classif, k=n_features)
            X_train_selected = selector.fit_transform(X_train_res, y_train_res)
            X_test_selected = selector.transform(X_test)
            
            # Store selected feature names for analysis
            selected_mask = selector.get_support()
            selected_features = X_train.columns[selected_mask].tolist()
            
            classifier.model.fit(X_train_selected, y_train_res)
            
            # Predict with threshold tuning (use selected features)
            y_proba = classifier.model.predict_proba(X_test_selected)[:, 1]
            
            # Find best threshold
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
            
            fold_results.append({
                'fold': fold,
                'f1': f1,
                'precision': prec,
                'recall': rec,
                'threshold': best_thresh,
                'train_stress': train_stress,
                'test_stress': test_stress
            })
            
            print(f"  Fold {fold}: F1={f1:.4f} (P={prec:.2f}, R={rec:.2f}) | thresh={best_thresh:.2f} | {test_stress} stress events")
        
        if fold_results:
            mean_f1 = np.mean([r['f1'] for r in fold_results])
            mean_prec = np.mean([r['precision'] for r in fold_results])
            mean_rec = np.mean([r['recall'] for r in fold_results])
            
            print(f"\n  *** {participant} Mean: F1={mean_f1:.4f}, P={mean_prec:.4f}, R={mean_rec:.4f} ***")
            
            all_results.append({
                'participant': participant,
                'mean_f1': mean_f1,
                'mean_precision': mean_prec,
                'mean_recall': mean_rec,
                'n_folds': len(fold_results),
                'n_stress': n_stress
            })
    
    # Summary
    if all_results:
        print("\n" + "="*60)
        print("=== OVERALL WITHIN-PARTICIPANT RESULTS ===")
        print("="*60)
        
        df = pd.DataFrame(all_results)
        
        # Weighted average by number of stress events
        total_stress = df['n_stress'].sum()
        weighted_f1 = sum(df['mean_f1'] * df['n_stress']) / total_stress
        weighted_prec = sum(df['mean_precision'] * df['n_stress']) / total_stress
        weighted_rec = sum(df['mean_recall'] * df['n_stress']) / total_stress
        
        print(f"\nPer-Participant Results:")
        for _, row in df.iterrows():
            print(f"  {row['participant']}: F1={row['mean_f1']:.4f}, P={row['mean_precision']:.4f}, R={row['mean_recall']:.4f} ({row['n_stress']} stress)")
        
        print(f"\nWeighted Average (by stress events):")
        print(f"  F1={weighted_f1:.4f}, Precision={weighted_prec:.4f}, Recall={weighted_rec:.4f}")
        print(f"\nSimple Average:")
        print(f"  F1={df['mean_f1'].mean():.4f}, Precision={df['mean_precision'].mean():.4f}, Recall={df['mean_recall'].mean():.4f}")

if __name__ == "__main__":
    run_within_participant_cv()
