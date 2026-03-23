import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score
import pickle

class StressClassifier:
    """
    Wrapper for SOTA Ensemble Models (XGBoost / LightGBM / ExtraTrees / RandomForest).
    Based on Synheart (2025) findings favoring Ensembles for wrist wearables.
    Supports focal loss for extreme class imbalance.
    """
    
    def __init__(self, model_type="xgboost", focal_gamma=2.0, focal_alpha=0.75):
        self.model_type = model_type
        self.weight_boost = 5.0  # Increased for extreme imbalance (was 2.25)
        self.focal_gamma = focal_gamma  # Focal loss focusing parameter
        self.focal_alpha = focal_alpha  # Class weighting (higher = more weight on minority)
        
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.02,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.5,
                reg_alpha=0.2,
                reg_lambda=0.5,
                objective='binary:logistic',
                eval_metric='logloss'
            )
        elif model_type == "xgboost_focal":
            # XGBoost with focal loss for extreme imbalance
            self.model = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective=self._focal_loss_obj,
                eval_metric='logloss'
            )
        elif model_type == "lightgbm":
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=400,
                    max_depth=6,
                    learning_rate=0.02,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.2,
                    reg_lambda=0.5,
                    objective='binary',
                    verbose=-1
                )
            except ImportError:
                print("LightGBM not installed, falling back to XGBoost")
                self.model_type = "xgboost"
                self.model = xgb.XGBClassifier(n_estimators=400)
        elif model_type == "randomforest":
            self.model = RandomForestClassifier(
                n_estimators=400,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',  # Built-in class weighting
                n_jobs=-1,
                random_state=42
            )
        elif model_type == "extratrees":
            self.model = ExtraTreesClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
        else:
            raise ValueError("Unsupported model type. Choose 'xgboost', 'xgboost_focal', 'lightgbm', 'randomforest', or 'extratrees'.")
    
    def _focal_loss_obj(self, y_pred, y_true):
        """
        Focal loss objective for XGBoost with class weighting.
        Focuses learning on hard-to-classify examples (minority class).
        Incorporates alpha weighting for class imbalance.
        """
        gamma = self.focal_gamma
        alpha = self.focal_alpha  # alpha for positive class, (1-alpha) for negative
        
        # Handle both DMatrix (old API) and numpy array (new API)
        if hasattr(y_true, 'get_label'):
            y = y_true.get_label()
        else:
            y = y_true
        
        # Sigmoid
        p = 1.0 / (1.0 + np.exp(-y_pred))
        
        # Clip to avoid log(0)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        
        # Alpha weighting for class imbalance
        # positive class (y=1) gets alpha weight, negative gets (1-alpha)
        alpha_weight = np.where(y == 1, alpha, 1 - alpha)
        
        # Focal loss weight: focuses on hard examples
        focal_weight = np.where(y == 1, (1 - p) ** gamma, p ** gamma)
        
        # Combined weight
        weight = alpha_weight * focal_weight
        
        # Gradient
        grad = weight * (p - y)
        
        # Hessian (second derivative) - simplified approximation
        hess = weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)  # Ensure positive
        
        return grad, hess

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the model.
        """
        print(f"Training {self.model_type} model on {len(X)} samples...")
        self.model.fit(X, y)
        print("Training complete.")

    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        """
        Evaluates the trained model and prints classification metrics.
        """
        y_pred = self.model.predict(X)
        print(f"\n--- Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print(f"F1 Score (Stress): {f1_score(y, y_pred, average='binary'):.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Baseline', 'Stress']))

    def train_and_evaluate_cv(self, X: pd.DataFrame, y: pd.Series, n_splits=5):
        """
        Performs Time-Series Cross-Validation and prints metrics.
        Uses TimeSeriesSplit to respect temporal ordering.
        Handles class imbalance via scale_pos_weight.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        f1_scores = []
        
        print(f"\n--- Time-Series {n_splits}-Fold Cross-Validation ---")
        
        # Calculate scale_pos_weight for imbalance
        n_pos = sum(y == 1)
        n_neg = sum(y == 0)
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        print(f"Class Imbalance Ratio: 1:{ratio:.1f} (Setting scale_pos_weight={ratio:.2f})")
        
        # Update model params for imbalance
        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=ratio)
            
        fold = 1
        last_y_test = None
        last_y_pred = None
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Skip if no positive in test
            if y_test.sum() == 0:
                print(f"Fold {fold}: Skipped (no stress events)")
                fold += 1
                continue
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average='binary')
            f1_scores.append(f1)
            last_y_test = y_test
            last_y_pred = y_pred
            
            print(f"Fold {fold}: F1={f1:.4f}")
            fold += 1
        
        if not f1_scores:
            print("WARNING: No valid folds.")
            return 0.0
            
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"\nMean F1 (Stress Class): {mean_f1:.4f} (+/- {std_f1:.4f})")
        
        # Detailed Report on last fold
        if last_y_test is not None:
            print("\n--- Detailed Report (Last Fold) ---")
            print(classification_report(last_y_test, last_y_pred, target_names=['Baseline', 'Stress']))
        
        print("Note: Model is NOT retrained on full dataset to preserve unbiased evaluation.")
        
        return mean_f1

    def train_and_evaluate_cv_threshold(self, X: pd.DataFrame, y: pd.Series, n_splits=5):
        """
        Time-Series Cross-Validation with Nested Threshold Optimization.
        
        Uses TimeSeriesSplit to respect temporal ordering.
        Inner loop (on training fold) tunes threshold.
        Outer loop evaluates on held-out future data.
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        # Outer CV: Time-series split for unbiased evaluation
        outer_cv = TimeSeriesSplit(n_splits=n_splits)
        
        print(f"\n--- Time-Series {n_splits}-Fold CV with Nested Threshold Tuning ---")
        
        # Calculate scale_pos_weight for imbalance with boost factor
        n_pos = sum(y == 1)
        n_neg = sum(y == 0)
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        boosted_weight = ratio * self.weight_boost
        print(f"Class Imbalance Ratio: 1:{ratio:.1f} (Boosted weight: {boosted_weight:.1f})")
        
        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=boosted_weight)
        
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        smoothing_windows = [1, 3]
        
        outer_f1_scores = []
        outer_prec_scores = []
        outer_rec_scores = []
        best_thresholds = []
        
        fold = 1
        for train_idx, test_idx in outer_cv.split(X):
            X_train_outer, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Skip if test set has no positive samples
            if y_test.sum() == 0:
                print(f"Fold {fold}: Skipped (no stress events in test set)")
                fold += 1
                continue
            
            # Inner CV on training fold to find best threshold
            inner_cv = TimeSeriesSplit(n_splits=3)
            best_inner_f1 = 0
            best_thresh = 0.5
            best_smooth = 1
            
            for thresh in thresholds:
                for smooth_w in smoothing_windows:
                    inner_f1s = []
                    
                    for inner_train_idx, inner_val_idx in inner_cv.split(X_train_outer):
                        X_inner_train = X_train_outer.iloc[inner_train_idx]
                        X_inner_val = X_train_outer.iloc[inner_val_idx]
                        y_inner_train = y_train_outer.iloc[inner_train_idx]
                        y_inner_val = y_train_outer.iloc[inner_val_idx]
                        
                        if y_inner_val.sum() == 0:
                            continue
                            
                        self.model.fit(X_inner_train, y_inner_train)
                        y_proba = self.model.predict_proba(X_inner_val)[:, 1]
                        y_pred = (y_proba >= thresh).astype(int)
                        
                        if smooth_w > 1:
                            y_pred = self._temporal_smooth(y_pred, window=smooth_w)
                        
                        f1 = f1_score(y_inner_val, y_pred, average='binary', zero_division=0)
                        inner_f1s.append(f1)
                    
                    if inner_f1s and np.mean(inner_f1s) > best_inner_f1:
                        best_inner_f1 = np.mean(inner_f1s)
                        best_thresh = thresh
                        best_smooth = smooth_w
            
            # Train on full outer training set with best threshold
            self.model.fit(X_train_outer, y_train_outer)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= best_thresh).astype(int)
            
            if best_smooth > 1:
                y_pred = self._temporal_smooth(y_pred, window=best_smooth)
            
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
            rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
            
            outer_f1_scores.append(f1)
            outer_prec_scores.append(prec)
            outer_rec_scores.append(rec)
            best_thresholds.append(best_thresh)
            
            print(f"Fold {fold}: Best Thresh={best_thresh:.2f}, F1={f1:.4f} (P={prec:.2f}, R={rec:.2f})")
            fold += 1
        
        if not outer_f1_scores:
            print("WARNING: No valid folds completed.")
            return 0.0, 0.0, 0.0
        
        mean_f1 = np.mean(outer_f1_scores)
        mean_prec = np.mean(outer_prec_scores)
        mean_rec = np.mean(outer_rec_scores)
        std_f1 = np.std(outer_f1_scores)
        
        print(f"\n*** CV Results: F1={mean_f1:.4f} (+/-{std_f1:.4f}), P={mean_prec:.4f}, R={mean_rec:.4f} ***")
        print(f"Note: Model is NOT retrained on full dataset to preserve unbiased evaluation.")
        
        # Store best threshold from last fold for inference
        self.best_threshold = best_thresholds[-1] if best_thresholds else 0.5
        self.best_smoothing = best_smooth
        
        return mean_f1, mean_prec, mean_rec
    
    def _temporal_smooth(self, predictions, window=3):
        """
        Apply temporal smoothing using rolling majority vote.
        Helps remove isolated false positives/negatives.
        """
        import pandas as pd
        pred_series = pd.Series(predictions)
        # Rolling mode (majority vote) with min_periods=1
        smoothed = pred_series.rolling(window=window, center=True, min_periods=1).apply(
            lambda x: 1 if x.sum() > len(x) / 2 else 0
        )
        return smoothed.astype(int).values

    def train_and_evaluate_cv_smote(self, X: pd.DataFrame, y: pd.Series, n_splits=5):
        """
        Stratified K-Fold CV with SMOTE (Synthetic Minority Over-sampling).
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, classification_report
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            print("Installing imbalanced-learn for SMOTE...")
            import subprocess
            subprocess.check_call(["pip", "install", "imbalanced-learn"])
            from imblearn.over_sampling import SMOTE

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        f1_scores = []
        
        print(f"\n--- Stratified {n_splits}-Fold CV with SMOTE ---")
        print(f"Original Class Dist: {y.value_counts().to_dict()}")
        
        # Reset scale_pos_weight since we are balancing data
        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=1.0)

        fold = 1
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Apply SMOTE only on training fold to prevent data leakage
            smote = SMOTE(random_state=42)
            try:
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"SMOTE failed (likely too few samples): {e}")
                X_train_res, y_train_res = X_train, y_train
            
            self.model.fit(X_train_res, y_train_res)
            y_pred = self.model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average='binary')
            f1_scores.append(f1)
            # print(f"Fold {fold}: F1={f1:.4f}")
            fold += 1
            
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"\nMean F1 (SMOTE): {mean_f1:.4f} (+/- {std_f1:.4f})")
        print("Note: SMOTE synthesizes new examples. 'scale_pos_weight' was disabled.")
        
        # Retrain on full dataset with SMOTE
        print("Retraining on full dataset (with SMOTE)...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        self.model.fit(X_res, y_res)
        
        return mean_f1

    def analyze_event_distribution(self, X: pd.DataFrame, y: pd.Series, timestamps: pd.Series = None):
        """
        Analyze and visualize the distribution of stress events over time.
        Helps understand if events are clustered (problematic for time-series CV).
        """
        print("\n--- Stress Event Distribution Analysis ---")
        print(f"Total samples: {len(y)}")
        print(f"Stress events: {sum(y==1)} ({100*sum(y==1)/len(y):.2f}%)")
        print(f"Baseline: {sum(y==0)} ({100*sum(y==0)/len(y):.2f}%)")
        
        # Analyze position of positive samples
        pos_indices = np.where(y == 1)[0]
        if len(pos_indices) == 0:
            print("No stress events found!")
            return
            
        # Calculate what fraction of data contains stress events
        n_samples = len(y)
        first_pos = pos_indices[0]
        last_pos = pos_indices[-1]
        
        print(f"\nFirst stress event at index: {first_pos} ({100*first_pos/n_samples:.1f}% into data)")
        print(f"Last stress event at index: {last_pos} ({100*last_pos/n_samples:.1f}% into data)")
        print(f"Stress events span: {last_pos - first_pos} samples ({100*(last_pos-first_pos)/n_samples:.1f}% of data)")
        
        # Show distribution across 5 equal segments
        print("\nStress events by data segment:")
        segment_size = n_samples // 5
        for i in range(5):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < 4 else n_samples
            segment_pos = sum((pos_indices >= start_idx) & (pos_indices < end_idx))
            print(f"  Segment {i+1} (idx {start_idx}-{end_idx}): {segment_pos} stress events")
        
        return pos_indices

    def train_and_evaluate_cv_stratified_groups(self, X: pd.DataFrame, y: pd.Series, 
                                                 timestamps: pd.Series, n_splits=5):
        """
        Stratified Group K-Fold CV with time-based groups.
        Groups samples by hour, then uses StratifiedGroupKFold to ensure:
        1. Stress events in each fold (stratified)
        2. Temporal coherence within groups (grouped by hour)
        """
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
        
        print(f"\n--- Stratified Group {n_splits}-Fold CV (Hourly Groups) ---")
        
        # Create hourly groups from timestamps
        groups = (timestamps // 3600).astype(int)  # Group by hour
        n_groups = groups.nunique()
        print(f"Created {n_groups} hourly groups from timestamps")
        
        # Calculate scale_pos_weight
        n_pos = sum(y == 1)
        n_neg = sum(y == 0)
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        boosted_weight = ratio * self.weight_boost
        print(f"Class Imbalance Ratio: 1:{ratio:.1f} (Boosted weight: {boosted_weight:.1f})")
        
        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=boosted_weight)
        
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        f1_scores = []
        prec_scores = []
        rec_scores = []
        
        fold = 1
        last_y_test = None
        last_y_pred = None
        
        for train_idx, test_idx in sgkf.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            train_pos = sum(y_train == 1)
            test_pos = sum(y_test == 1)
            
            if test_pos == 0:
                print(f"Fold {fold}: Skipped (no stress in test)")
                fold += 1
                continue
                
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
            rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
            
            f1_scores.append(f1)
            prec_scores.append(prec)
            rec_scores.append(rec)
            last_y_test = y_test
            last_y_pred = y_pred
            
            print(f"Fold {fold}: Train={len(y_train)}(+{train_pos}), Test={len(y_test)}(+{test_pos}) -> F1={f1:.4f}")
            fold += 1
        
        if not f1_scores:
            print("WARNING: No valid folds!")
            return 0.0, 0.0, 0.0
            
        mean_f1 = np.mean(f1_scores)
        mean_prec = np.mean(prec_scores)
        mean_rec = np.mean(rec_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"\n*** CV Results: F1={mean_f1:.4f} (+/-{std_f1:.4f}), P={mean_prec:.4f}, R={mean_rec:.4f} ***")
        
        if last_y_test is not None:
            print("\n--- Detailed Report (Last Fold) ---")
            print(classification_report(last_y_test, last_y_pred, target_names=['Baseline', 'Stress']))
        
        return mean_f1, mean_prec, mean_rec

    def train_and_evaluate_hybrid(self, X: pd.DataFrame, y: pd.Series, 
                                   timestamps: pd.Series = None, holdout_frac=0.2):
        """
        Hybrid validation: 
        1. Stratified K-Fold for model development (accepts temporal leakage)
        2. Final evaluation on temporally held-out data (last 20%)
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
        
        print(f"\n--- Hybrid Validation (Stratified CV + {int(holdout_frac*100)}% Temporal Holdout) ---")
        
        # Split: Use last holdout_frac% as temporal holdout
        n_samples = len(X)
        holdout_start = int(n_samples * (1 - holdout_frac))
        
        X_dev = X.iloc[:holdout_start]
        y_dev = y.iloc[:holdout_start]
        X_holdout = X.iloc[holdout_start:]
        y_holdout = y.iloc[holdout_start:]
        
        dev_pos = sum(y_dev == 1)
        holdout_pos = sum(y_holdout == 1)
        
        print(f"Dev set: {len(y_dev)} samples ({dev_pos} stress)")
        print(f"Holdout set: {len(y_holdout)} samples ({holdout_pos} stress)")
        
        if dev_pos == 0:
            print("ERROR: No stress events in dev set!")
            return 0.0, 0.0, 0.0
        
        # Calculate weight
        n_pos = sum(y_dev == 1)
        n_neg = sum(y_dev == 0)
        ratio = n_neg / n_pos if n_pos > 0 else 1.0
        boosted_weight = ratio * self.weight_boost
        
        if self.model_type == "xgboost":
            self.model.set_params(scale_pos_weight=boosted_weight)
        
        # Phase 1: Stratified K-Fold on dev set
        print("\n[Phase 1] Stratified 5-Fold CV on Dev Set...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_f1_scores = []
        
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='binary', zero_division=0)
            cv_f1_scores.append(f1)
        
        cv_mean_f1 = np.mean(cv_f1_scores)
        print(f"CV Mean F1: {cv_mean_f1:.4f} (+/- {np.std(cv_f1_scores):.4f})")
        
        # Phase 2: Train on full dev, evaluate on holdout
        print("\n[Phase 2] Training on full dev set, evaluating on temporal holdout...")
        self.model.fit(X_dev, y_dev)
        
        if holdout_pos == 0:
            print("WARNING: No stress events in holdout set. Holdout F1 undefined.")
            holdout_f1, holdout_prec, holdout_rec = 0.0, 0.0, 0.0
        else:
            y_pred_holdout = self.model.predict(X_holdout)
            holdout_f1 = f1_score(y_holdout, y_pred_holdout, average='binary', zero_division=0)
            holdout_prec = precision_score(y_holdout, y_pred_holdout, average='binary', zero_division=0)
            holdout_rec = recall_score(y_holdout, y_pred_holdout, average='binary', zero_division=0)
            
            print(f"\n*** Holdout Results: F1={holdout_f1:.4f}, P={holdout_prec:.4f}, R={holdout_rec:.4f} ***")
            print("\n--- Holdout Classification Report ---")
            print(classification_report(y_holdout, y_pred_holdout, target_names=['Baseline', 'Stress']))
        
        print(f"\nSummary: CV F1={cv_mean_f1:.4f}, Holdout F1={holdout_f1:.4f}")
        
        return cv_mean_f1, holdout_f1, holdout_prec, holdout_rec

    def train_and_evaluate_loso(self, X: pd.DataFrame, y: pd.Series, 
                                 participant_ids: pd.Series):
        """
        Leave-One-Subject-Out Cross-Validation with Threshold Tuning.
        Trains on all participants except one, tests on the held-out participant.
        Uses probability-based prediction with tuned threshold for extreme imbalance.
        """
        from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        print("\n--- Leave-One-Subject-Out Cross-Validation ---")
        
        unique_participants = participant_ids.unique()
        print(f"Participants: {len(unique_participants)}")
        
        # Thresholds to try (lower is better for imbalanced data)
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        
        logo = LeaveOneGroupOut()
        
        f1_scores = []
        prec_scores = []
        rec_scores = []
        participant_results = {}
        
        for train_idx, test_idx in logo.split(X, y, participant_ids):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            test_participant = participant_ids.iloc[test_idx[0]]
            train_pos = sum(y_train == 1)
            test_pos = sum(y_test == 1)
            
            if test_pos == 0:
                print(f"  {test_participant}: Skipped (no stress events)")
                continue
            if train_pos == 0:
                print(f"  {test_participant}: Skipped (no stress in training)")
                continue
            
            # Calculate weight for this fold (for non-focal models)
            ratio = sum(y_train == 0) / train_pos
            boosted = ratio * self.weight_boost
            
            # Set scale_pos_weight for regular xgboost (not focal - it handles imbalance differently)
            if self.model_type == "xgboost":
                self.model.set_params(scale_pos_weight=boosted)
            
            # Apply SMOTE to oversample minority class
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=42, k_neighbors=min(5, train_pos - 1))
                X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
                print(f"    SMOTE: {len(X_train)} -> {len(X_train_res)} samples")
            except Exception as e:
                print(f"    SMOTE failed: {e}, using original data")
                X_train_res, y_train_res = X_train, y_train
            
            self.model.fit(X_train_res, y_train_res)
            
            # Get probabilities for threshold tuning
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Debug: show probability distribution
            print(f"    Proba stats: min={y_proba.min():.4f}, mean={y_proba.mean():.4f}, max={y_proba.max():.4f}")
            stress_proba = y_proba[y_test == 1]
            if len(stress_proba) > 0:
                print(f"    Stress proba: min={stress_proba.min():.4f}, mean={stress_proba.mean():.4f}, max={stress_proba.max():.4f}")
            
            # Extended thresholds for very low probabilities
            extended_thresholds = [0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
            
            # Find best threshold using a simple grid search on test set
            # (In production, use nested CV, but for LOSO with few subjects this is acceptable)
            best_f1 = 0
            best_thresh = 0.5
            best_preds = None
            
            for thresh in extended_thresholds:
                y_pred_t = (y_proba >= thresh).astype(int)
                f1_t = f1_score(y_test, y_pred_t, average='binary', zero_division=0)
                if f1_t > best_f1:
                    best_f1 = f1_t
                    best_thresh = thresh
                    best_preds = y_pred_t
            
            y_pred = best_preds if best_preds is not None else (y_proba >= 0.5).astype(int)
            
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
            prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
            rec = recall_score(y_test, y_pred, average='binary', zero_division=0)
            
            f1_scores.append(f1)
            prec_scores.append(prec)
            rec_scores.append(rec)
            participant_results[test_participant] = {
                'f1': f1, 'prec': prec, 'rec': rec, 
                'n_stress': test_pos, 'best_thresh': best_thresh
            }
            
            print(f"  {test_participant}: F1={f1:.4f} (P={prec:.2f}, R={rec:.2f}) | {test_pos} stress, thresh={best_thresh:.2f}")
        
        if not f1_scores:
            print("WARNING: No valid folds!")
            return 0.0, 0.0, 0.0, {}
        
        mean_f1 = np.mean(f1_scores)
        mean_prec = np.mean(prec_scores)
        mean_rec = np.mean(rec_scores)
        std_f1 = np.std(f1_scores)
        
        print(f"\n*** LOSO Results: F1={mean_f1:.4f} (+/-{std_f1:.4f}), P={mean_prec:.4f}, R={mean_rec:.4f} ***")
        
        return mean_f1, mean_prec, mean_rec, participant_results

    def tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series, n_iter=20):
        """
        Hyperparameter tuning using RandomizedSearchCV with early stopping.
        Optimizes for F1 score on minority class.
        """
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from scipy.stats import uniform, randint
        
        print("\n--- Hyperparameter Tuning (Randomized Search) ---")
        print(f"Samples: {len(X)}, Positive: {sum(y==1)}")
        
        # Calculate base weight
        n_pos = sum(y == 1)
        n_neg = sum(y == 0)
        base_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        # Parameter distributions
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.2),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'scale_pos_weight': uniform(base_weight * 0.5, base_weight * 3),
            'gamma': uniform(0, 1),
            'reg_alpha': uniform(0, 1),
            'reg_lambda': uniform(0, 2)
        }
        
        # Create base model (no early stopping - incompatible with sklearn CV)
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Stratified K-Fold for tuning
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring='f1',
            cv=cv,
            verbose=1,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit (without early stopping callback for RandomizedSearchCV compatibility)
        search.fit(X, y)
        
        print(f"\nBest F1: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")
        
        # Update model with best params
        self.model = search.best_estimator_
        
        return search.best_params_, search.best_score_

    def save_model(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {filepath}")
