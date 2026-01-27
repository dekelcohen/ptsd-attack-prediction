"""
Optimized Training Script for Single Participant.

Uses:
- Parallel file processing
- Feature caching
- 75% overlap for more samples
- New cross-modal and derivative features
"""

from pipeline import StressDetectionPipeline
import os

# Configuration
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_FILE = r"D:\workdir\ptsd_stress_detection\refined_tags\TRAIL10_refined_tags_test.csv"
TARGET_PARTICIPANT = "TRAIL10"

def run_optimized():
    print("=== Optimized Single-Participant Training ===\n")
    
    # Initialize with parallel processing
    pipeline = StressDetectionPipeline(use_model="xgboost", n_jobs=-1)
    
    # Process with caching (loads from cache if available)
    features = pipeline.process_participant_cached(
        DATA_DIR, 
        TARGET_PARTICIPANT, 
        overlap_percent=0.75
    )
    
    if features.empty:
        print("No features extracted.")
        return
    
    print(f"\nExtracted {len(features)} feature windows")
    
    # Load and align labels
    tags = pipeline.load_labels(TAGS_FILE)
    labeled_data = pipeline.align_labels(features, tags)
    
    print(f"\n--- Dataset Summary ---")
    print(f"Windows: {len(labeled_data)}")
    print(f"Class Balance: {labeled_data['label'].value_counts().to_dict()}")
    
    if len(labeled_data['label'].unique()) <= 1:
        print("Dataset has only one class. Cannot train.")
        return
    
    # Prepare features
    X = labeled_data.drop(columns=['label', 'start_time', 'end_time', 'source_file', 'timestamp'], errors='ignore')
    y = labeled_data['label']
    timestamps = labeled_data['start_time'] if 'start_time' in labeled_data.columns else None
    
    # Show feature count
    print(f"Features: {len(X.columns)}")
    print(f"New features: {[c for c in X.columns if 'ratio' in c or 'deriv' in c or 'product' in c]}")
    
    # 1. Analyze event distribution
    pipeline.classifier.analyze_event_distribution(X, y)
    
    # 2. Stratified Group K-Fold (best for single participant)
    if timestamps is not None:
        pipeline.classifier.train_and_evaluate_cv_stratified_groups(X, y, timestamps)
    
    # 3. Hybrid validation
    pipeline.classifier.train_and_evaluate_hybrid(X, y, timestamps)

def run_with_tuning():
    """Run with hyperparameter tuning."""
    print("=== Optimized Training with Hyperparameter Tuning ===\n")
    
    pipeline = StressDetectionPipeline(use_model="xgboost", n_jobs=-1)
    
    features = pipeline.process_participant_cached(
        DATA_DIR, 
        TARGET_PARTICIPANT, 
        overlap_percent=0.75
    )
    
    if features.empty:
        print("No features extracted.")
        return
    
    tags = pipeline.load_labels(TAGS_FILE)
    labeled_data = pipeline.align_labels(features, tags)
    
    if len(labeled_data['label'].unique()) <= 1:
        print("Dataset has only one class.")
        return
    
    X = labeled_data.drop(columns=['label', 'start_time', 'end_time', 'source_file', 'timestamp'], errors='ignore')
    y = labeled_data['label']
    timestamps = labeled_data['start_time'] if 'start_time' in labeled_data.columns else None
    
    # Tune hyperparameters
    best_params, best_score = pipeline.classifier.tune_hyperparameters(X, y, n_iter=15)
    
    # Evaluate with tuned model (don't override params)
    print("\n--- Evaluating Tuned Model with Stratified Group CV ---")
    print("Note: Using tuned hyperparameters, not resetting scale_pos_weight")
    
    if timestamps is not None:
        # Use the tuned model directly without resetting params
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        groups = (timestamps // 3600).astype(int)
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        
        f1_scores = []
        for train_idx, test_idx in sgkf.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            if y_test.sum() == 0:
                continue
            
            pipeline.classifier.model.fit(X_train, y_train)
            y_pred = pipeline.classifier.model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            f1_scores.append(f1)
            print(f"  Fold: F1={f1:.4f}")
        
        if f1_scores:
            import numpy as np
            print(f"\n*** Tuned Model CV Results: F1={np.mean(f1_scores):.4f} (+/-{np.std(f1_scores):.4f}) ***")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--tune":
        run_with_tuning()
    else:
        run_optimized()
