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
import argparse
import json

# Configuration
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
# Use v2 refined tags with confidence scoring and physiological validation
TAGS_DIR = r"D:\workdir\ptsd-attack-prediction\ptsd_stress_detection\refined_tags"
PERIODS_FILE = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\participant_data_periods.xlsx"

PARTICIPANTS = ["TRAIL009", "TRAIL10", "TRAIL003", "TRAIL006", "TRAIL013"]

WINDOW_SIZE_MIN = 10
N_SPLITS = 5
BUFFER_BEFORE_MIN = 8
BUFFER_AFTER_MIN = 15

# Parse arguments
MODEL_TYPE = "xgboost"
USE_COMBINED = False

for arg in sys.argv[1:]:
    if arg == "--combined":
        USE_COMBINED = True
    elif arg in ["xgboost", "lightgbm", "randomforest"]:
        MODEL_TYPE = arg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?", default=MODEL_TYPE, choices=["xgboost", "lightgbm", "randomforest"])
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--buffer-before", type=int, default=BUFFER_BEFORE_MIN)
    parser.add_argument("--buffer-after", type=int, default=BUFFER_AFTER_MIN)
    parser.add_argument("--splits", type=int, default=N_SPLITS)
    parser.add_argument("--window-min", type=int, default=WINDOW_SIZE_MIN)
    parser.add_argument("--participants", type=str, default=",".join(PARTICIPANTS))
    parser.add_argument("--participants-from-periods", action="store_true")
    parser.add_argument("--neg-pos-ratio", type=float, default=0.0,
                        help="If >0, downsample training negatives to this ratio per positive in each fold")
    parser.add_argument("--results-file", type=str, default="")
    return parser.parse_args()


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

def add_rolling_lag_features(X: pd.DataFrame, window_sizes=[3, 5]) -> pd.DataFrame:
    """Add rolling statistics and lag features."""
    X_new = X.copy()
    key_features = ['eda_phasic_mean', 'eda_scr_freq', 'hr_mean', 'hr_rmssd', 'temp_mean', 'acc_mean']
    existing_keys = [f for f in key_features if f in X.columns]
    
    for feature in existing_keys:
        for window in window_sizes:
            X_new[f'{feature}_roll_mean_{window}'] = X[feature].shift(1).rolling(window=window, min_periods=1).mean()
            X_new[f'{feature}_roll_std_{window}'] = X[feature].shift(1).rolling(window=window, min_periods=1).std()
            X_new[f'{feature}_lag_{window}'] = X[feature].shift(window)
    
    X_new = X_new.ffill().bfill()
    return X_new


def apply_hysteresis(y_proba: np.ndarray, high_thresh: float, low_thresh: float, min_on: int = 1) -> np.ndarray:
    preds = np.zeros_like(y_proba, dtype=int)
    state_on = False
    pending = 0

    for idx, prob in enumerate(y_proba):
        if state_on:
            if prob < low_thresh:
                state_on = False
            else:
                preds[idx] = 1
                continue

        if prob >= high_thresh:
            pending += 1
            if pending >= min_on:
                state_on = True
                start_idx = max(0, idx - min_on + 1)
                preds[start_idx:idx + 1] = 1
        else:
            pending = 0

    return preds


def tune_decision_rule(y_val: pd.Series, y_val_proba: np.ndarray):
    best = {
        'f1': 0.0,
        'high': 0.5,
        'low': 0.45,
        'min_on': 1,
    }

    for high in np.arange(0.40, 0.96, 0.05):
        for gap in [0.0, 0.05, 0.10]:
            low = max(0.05, high - gap)
            for min_on in [1, 2, 3]:
                y_pred = apply_hysteresis(y_val_proba, high_thresh=float(high), low_thresh=float(low), min_on=min_on)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                if f1 > best['f1']:
                    best = {'f1': f1, 'high': float(high), 'low': float(low), 'min_on': int(min_on)}

    if best['f1'] == 0.0:
        fallback = {
            'f1': 0.0,
            'high': 0.20,
            'low': 0.15,
            'min_on': 1,
            'recall': -1.0,
            'precision': 0.0,
        }
        for high in np.arange(0.05, 0.56, 0.05):
            for gap in [0.0, 0.05, 0.10]:
                low = max(0.01, high - gap)
                for min_on in [1, 2, 3]:
                    y_pred = apply_hysteresis(y_val_proba, high_thresh=float(high), low_thresh=float(low), min_on=min_on)
                    rec = recall_score(y_val, y_pred, zero_division=0)
                    prec = precision_score(y_val, y_pred, zero_division=0)
                    f1 = f1_score(y_val, y_pred, zero_division=0)
                    if rec > fallback['recall'] and prec >= 0.05:
                        fallback = {'f1': f1, 'high': float(high), 'low': float(low), 'min_on': int(min_on), 'recall': rec, 'precision': prec}

        if fallback['recall'] < 0:
            for high in np.arange(0.05, 0.56, 0.05):
                for gap in [0.0, 0.05, 0.10]:
                    low = max(0.01, high - gap)
                    for min_on in [1, 2, 3]:
                        y_pred = apply_hysteresis(y_val_proba, high_thresh=float(high), low_thresh=float(low), min_on=min_on)
                        rec = recall_score(y_val, y_pred, zero_division=0)
                        prec = precision_score(y_val, y_pred, zero_division=0)
                        f1 = f1_score(y_val, y_pred, zero_division=0)
                        if rec > fallback['recall']:
                            fallback = {'f1': f1, 'high': float(high), 'low': float(low), 'min_on': int(min_on), 'recall': rec, 'precision': prec}

        best = {'f1': fallback['f1'], 'high': fallback['high'], 'low': fallback['low'], 'min_on': fallback['min_on']}

    return best

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
    # Look for v2 refined tags first
    pattern_v2 = os.path.join(TAGS_DIR, f"{participant_id}_refined_tags_v2.csv")
    if os.path.exists(pattern_v2):
        return pattern_v2
    
    # Fallback to any matching pattern
    pattern = os.path.join(TAGS_DIR, f"{participant_id}*.csv")
    files = glob.glob(pattern)
    if files:
        return max(files, key=os.path.getmtime)
    return None

def train_and_evaluate(X_train, y_train, X_test, y_test, classifier, fold_name, sample_weight_train=None, neg_pos_ratio: float = 0.0):
    """Train and evaluate a single fold."""
    train_stress = sum(y_train == 1)
    test_stress = sum(y_test == 1)
    
    if train_stress == 0 or test_stress == 0:
        return None
    
    if sample_weight_train is None:
        sample_weight_train = np.ones(len(y_train), dtype=float)
    else:
        sample_weight_train = np.asarray(sample_weight_train, dtype=float)

    ratio = sum(y_train == 0) / train_stress

    if neg_pos_ratio and neg_pos_ratio > 0:
        positive_idx = np.where(y_train.values == 1)[0]
        negative_idx = np.where(y_train.values == 0)[0]
        max_neg = int(len(positive_idx) * neg_pos_ratio)
        if len(positive_idx) > 0 and len(negative_idx) > max_neg and max_neg > 0:
            rng = np.random.default_rng(42)
            kept_negative_idx = rng.choice(negative_idx, size=max_neg, replace=False)
            keep_idx = np.sort(np.concatenate([positive_idx, kept_negative_idx]))
            X_train = X_train.iloc[keep_idx].reset_index(drop=True)
            y_train = y_train.iloc[keep_idx].reset_index(drop=True)
            sample_weight_train = sample_weight_train[keep_idx]
            train_stress = int(sum(y_train == 1))
            ratio = sum(y_train == 0) / max(1, train_stress)

    val_size = max(1, int(len(X_train) * 0.2))
    if len(X_train) < 40:
        val_size = max(1, int(len(X_train) * 0.25))

    X_inner_train = X_train.iloc[:-val_size] if val_size < len(X_train) else X_train
    y_inner_train = y_train.iloc[:-val_size] if val_size < len(y_train) else y_train
    w_inner_train = sample_weight_train[:-val_size] if val_size < len(sample_weight_train) else sample_weight_train

    X_val = X_train.iloc[-val_size:] if val_size < len(X_train) else X_train
    y_val = y_train.iloc[-val_size:] if val_size < len(y_train) else y_train

    n_features = min(35, X_inner_train.shape[1])
    selector = SelectKBest(mutual_info_classif, k=n_features)
    X_inner_selected = selector.fit_transform(X_inner_train, y_inner_train)
    X_val_selected = selector.transform(X_val)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    if hasattr(classifier.model, 'set_params') and classifier.model_type == "xgboost":
        classifier.model.set_params(scale_pos_weight=ratio * 3.0)
    if hasattr(classifier.model, 'set_params') and classifier.model_type == "lightgbm":
        classifier.model.set_params(scale_pos_weight=ratio * 1.8)
    
    if y_inner_train.nunique() < 2:
        return None

    classifier.model.fit(X_inner_selected, y_inner_train, sample_weight=w_inner_train)
    val_proba = classifier.model.predict_proba(X_val_selected)
    if val_proba.ndim != 2 or val_proba.shape[1] < 2:
        return None
    y_val_proba = val_proba[:, 1]
    best_rule = tune_decision_rule(y_val, y_val_proba)

    if y_train.nunique() < 2:
        return None

    classifier.model.fit(X_train_selected, y_train, sample_weight=sample_weight_train)
    test_proba = classifier.model.predict_proba(X_test_selected)
    if test_proba.ndim != 2 or test_proba.shape[1] < 2:
        return None
    y_proba = test_proba[:, 1]
    y_pred = apply_hysteresis(
        y_proba,
        high_thresh=best_rule['high'],
        low_thresh=best_rule['low'],
        min_on=best_rule['min_on']
    )

    f1 = f1_score(y_test, y_pred, zero_division=0)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    
    print(f"  {fold_name}: F1={f1:.4f} (P={prec:.2f}, R={rec:.2f}) | high={best_rule['high']:.2f}, low={best_rule['low']:.2f}, on={best_rule['min_on']} | {test_stress} stress")
    
    return {'f1': f1, 'precision': prec, 'recall': rec, 'threshold': best_rule['high'], 'test_stress': test_stress}

def run_single_participant_cv(participant, pipeline, all_data, buffer_before_min=10, buffer_after_min=20, n_splits=5, neg_pos_ratio: float = 0.0):
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
        
    # Normalize features per participant (Z-Score)
    features = pipeline.normalize_features(features)
    
    
    # Filter High Activity (Physical Load) - DISABLED (Lowered F1 from 0.54 to 0.51)
    # Analysis showed some stress events occur during activity (e.g. TRAIL006)
    # if 'acc_mean' in features.columns:
    #     initial_len = len(features)
    #     features = features[features['acc_mean'] < 5.0] # Relaxed or removed
    #     dropped = initial_len - len(features)
    #     if dropped > 0:
    #         print(f"  Filtered {dropped} high-activity windows.")
    
    tags = pipeline.load_labels(tags_file)
    labeled = pipeline.align_labels(features, tags, buffer_before_min=buffer_before_min, buffer_after_min=buffer_after_min)
    
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
    sample_weights = labeled['label_weight'].values if 'label_weight' in labeled.columns else np.ones(len(labeled), dtype=float)
    
    # Add rolling/lag features
    X = add_rolling_lag_features(X)
    
    # Store for combined training
    labeled['participant_id'] = participant
    all_data.append(labeled)
    
    # Check if stress is too clustered for TimeSeriesSplit
    use_stratified = analyze_stress_clustering(y)
    
    if use_stratified:
        print(f"  ⚠️ Stress clustered - using Stratified Shuffle-Split instead of TimeSeries")
        cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        cv_iter = cv.split(X, y)
    else:
        cv = TimeSeriesSplit(n_splits=n_splits)
        cv_iter = cv.split(X)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(cv_iter, 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weights[train_idx]
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                   pipeline.classifier, f"Fold {fold}", sample_weight_train=w_train,
                                   neg_pos_ratio=neg_pos_ratio)
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

def run_combined_training(all_data, pipeline, n_splits=5, neg_pos_ratio: float = 0.0):
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
    sample_weights = combined['label_weight'].values if 'label_weight' in combined.columns else np.ones(len(combined), dtype=float)
    
    X = add_rolling_lag_features(X)
    
    # Use StratifiedShuffleSplit for combined data
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        w_train = sample_weights[train_idx]
        
        result = train_and_evaluate(X_train, y_train, X_test, y_test, 
                                   pipeline.classifier, f"Combined Fold {fold}", sample_weight_train=w_train,
                                   neg_pos_ratio=neg_pos_ratio)
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
    args = parse_args()
    summary = {
        'model': args.model,
        'combined': bool(args.combined),
        'buffer_before': int(args.buffer_before),
        'buffer_after': int(args.buffer_after),
        'window_min': int(args.window_min),
        'neg_pos_ratio': float(args.neg_pos_ratio),
        'participants': [],
        'per_participant': [],
        'weighted': None,
        'combined_results': None,
    }

    print("=== Enhanced Within-Participant CV ===\n")
    print(f"Model: {args.model}")
    print(f"Combined Training: {args.combined}")
    print(f"Label Buffers: -{args.buffer_before}m / +{args.buffer_after}m")
    if args.participants_from_periods:
        participants = load_participants_from_periods(PERIODS_FILE)
        print(f"Participants loaded from periods file (excluding TRAIL007): {participants}")
    else:
        participants = [p.strip() for p in args.participants.split(',') if p.strip()]
    print(f"Participants: {participants}")
    summary['participants'] = participants
    
    pipeline = StressDetectionPipeline(use_model=args.model, n_jobs=-1, window_size_min=args.window_min)
    
    all_results = []
    all_data = []
    
    # Run per-participant CV
    for participant in participants:
        result, _ = run_single_participant_cv(
            participant,
            pipeline,
            all_data,
            buffer_before_min=args.buffer_before,
            buffer_after_min=args.buffer_after,
            n_splits=args.splits,
            neg_pos_ratio=args.neg_pos_ratio,
        )
        if result:
            all_results.append(result)
            summary['per_participant'].append(result)
    
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
        summary['weighted'] = {
            'f1': float(weighted_f1),
            'precision': float(weighted_prec),
            'recall': float(weighted_rec),
        }
    
    # Combined training
    if args.combined and len(all_data) >= 2:
        summary['combined_results'] = run_combined_training(all_data, pipeline, n_splits=args.splits, neg_pos_ratio=args.neg_pos_ratio)

    if args.results_file:
        with open(args.results_file, 'w', encoding='utf-8') as fp:
            json.dump(summary, fp, indent=2)
        print(f"Saved results to: {args.results_file}")

if __name__ == "__main__":
    main()
