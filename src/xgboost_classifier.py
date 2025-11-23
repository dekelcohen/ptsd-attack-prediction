import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import f1_score, precision_recall_curve

import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, ParameterGrid
from sklearn.metrics import classification_report

from data_preparation import prepare_biomarkers_data, create_chunked_data

import warnings
warnings.filterwarnings("ignore")


class xgboost_model():
    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, multiclassification, output_dir, standard_scale=False,
                 n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0):
        self.multiclassification = multiclassification
        self.output_dir = output_dir
        num_classes = set(pd.concat([y_train, y_test]))
        y_train_count = y_train.value_counts()
        self.scale_pos_weight = y_train_count[0] / y_train_count[1]
        if standard_scale:
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)
        self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = X_train, X_val, X_test, y_train, y_val, y_test

        if multiclassification:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,

                random_state=42,
                objective="multi:softprob ",  # or "multi:softmax"
                num_class=num_classes,
                scale_pos_weight=self.scale_pos_weight,
                eval_metric="mlogloss"
            )
        else:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,

                random_state=42,
                objective="binary:logistic",
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=self.scale_pos_weight
            )

        self.threshold = 0.5

    def run_model(self, grid_search=False):
        if grid_search:
            best_model, best_info, results_df = self.run_grid_search()

            print("Best params:", best_info["params"])
            print("Best val F1:", round(best_info["val_f1"], 4))
            print("Best threshold:", round(best_info["threshold"], 4))
            self.model = best_model
        else:
            self.model.fit(self.X_train, self.y_train)

        y_category_pred = self.model.predict(self.X_val)
        print("\n--- Initial Classification Report (with 0.5 threshold) ---")
        print(classification_report(self.y_val, y_category_pred, target_names=['No Stress', 'Stress']))

        # Get the probability scores for the positive class (Stress)
        y_pred = self.model.predict_proba(self.X_val)[:, 1]

        # --- Plot the Precision-Recall Curve ---
        display = PrecisionRecallDisplay.from_predictions(self.y_val, y_pred, name="XGBoost")
        display.plot()
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(self.output_dir, "precision_recall_curve.png"), dpi=300, bbox_inches="tight")

        self.threshold = self.calculate_best_threshold(y_pred)

        print(f"Best Threshold based on F1-Score: {self.threshold:.4f}")

        # 3. Apply the new threshold to your probability scores
        new_predictions = (y_pred >= self.threshold).astype(int)

        # 4. Compare the new classification report with the old one
        print(f"\n--- New Threshold Classification Report (with custom threshold {self.threshold}) ---")
        report = classification_report(self.y_val, new_predictions, target_names=['No Stress', 'Stress'], digits=4)
        with open(f"{self.output_dir}/classification_report.txt", "w") as f:
            f.write(report)
        print(report)

        self.gen_confusion_matrix(new_predictions, self.y_val)
        # self.plot_important_features()
        self.eval_on_test()

        print(self.output_dir)

    def eval_on_test(self):
        y_pred = self.model.predict_proba(self.X_test)[:, 1]
        new_predictions = (y_pred >= self.threshold).astype(int)

        print(f"\n--- New Threshold Classification Report on Test Set (with custom threshold {self.threshold}) ---")
        report = classification_report(self.y_test, new_predictions, target_names=['No Stress', 'Stress'], digits=4)
        with open(f"{self.output_dir}/test_set_classification_report.txt", "w") as f:
            f.write(report)
        print(report)

        self.gen_confusion_matrix(new_predictions, self.y_test, title='Confusion matrix on test set')

    def calculate_best_threshold(self, y_pred):
        # 1. Get precision, recall, and thresholds
        precision, recall, thresholds = precision_recall_curve(self.y_val, y_pred)

        # 2. Find the threshold that maximizes the F1-score
        # F1 = 2 * (precision * recall) / (precision + recall)
        # We add a small number (1e-9) to avoid division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

        # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold.
        # So we slice the F1 scores to match the number of thresholds.
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        return best_threshold

    def gen_confusion_matrix(self, new_predictions, org_values, title='Confusion matrix'):
        cm = confusion_matrix(org_values, new_predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(title)
        plt.savefig(os.path.join(self.output_dir, f"{title}.png"), dpi=300, bbox_inches="tight")
        plt.close()

    def plot_important_features(self):
        plt.figure(figsize=(10, 8))  # adjust width/height as needed
        xgb.plot_importance(self.model, max_num_features=15)

        plt.tight_layout()  # makes sure labels and titles fit into the figure
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), dpi=300)
        plt.close()

    def run_grid_search(self, tune_threshold=True):
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            # 'gamma': [0, 1, 5],
            # 'min_child_weight': [1, 3, 5],
            # 'scale_pos_weight': [1, 2, 5]  # useful for imbalanced datasets
        }

        results = []
        best = {"val_f1": -np.inf}

        for params in ParameterGrid(param_grid):
            clf = xgb.XGBClassifier(
                objective="binary:logistic",
                random_state=42,
                scale_pos_weight=self.scale_pos_weight,
                **params
            )

            fit_kwargs = {}
            # if early_stopping_rounds is not None:
            #     fit_kwargs.update(
            #         dict(
            #             eval_set=[(X_val, y_val)],
            #             eval_metric=eval_metric,
            #             early_stopping_rounds=early_stopping_rounds,
            #             verbose=False,
            #         )
            #     )

            clf.fit(self.X_train, self.y_train, **fit_kwargs)

            # Probabilities for threshold sweeping
            y_proba = clf.predict_proba(self.X_val)[:, 1]

            if tune_threshold:
                prec, rec, thr = precision_recall_curve(self.y_val, y_proba, pos_label=1)
                # precision_recall_curve returns one more P/R point than thresholds
                f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
                idx = int(np.nanargmax(f1s))
                val_f1 = float(f1s[idx])
                best_thr = float(thr[idx])
            else:
                best_thr = 0.5
                val_f1 = f1_score(y_val, (y_proba >= best_thr).astype(int), pos_label=1)

            results.append(
                {"params": params, "val_f1": val_f1, "best_threshold": best_thr,
                 "n_estimators": clf.get_params().get("n_estimators")}
            )

            if val_f1 > best["val_f1"]:
                best = {"params": params, "val_f1": val_f1, "threshold": best_thr, "model": clf}

        results_df = pd.DataFrame(results).sort_values("val_f1", ascending=False).reset_index(drop=True)
        return best["model"], {"params": best["params"], "val_f1": best["val_f1"],
                               "threshold": best["threshold"]}, results_df


def split_train_test_by_day(X):
    unique_days = X["timestamp_israel"].dt.date.unique()

    # Split based on days
    train_days, val_days = train_test_split(
        unique_days, test_size=0.25, random_state=40
    )

    # Create masks for samples belonging to the selected days
    train_mask = X["timestamp_israel"].dt.date.isin(train_days)
    test_mask = X["timestamp_israel"].dt.date.isin(val_days)

    # Build train/validation sets
    X_train, X_test = X[train_mask], X[test_mask]

    return X_train, X_test


def prep_onehotencoded_columns(orgX):
    eventType_classes = [x for x in list(set(orgX['eventType'])) if pd.notna(x)]
    # activity_intensity_classes = [x for x in list(set(orgX['activity_intensity'])) if pd.notna(x)]
    # body_position_left_classes = [x for x in list(set(orgX['body_position_left'])) if pd.notna(x)]
    # body_position_right_classes = [x for x in list(set(orgX['body_position_right'])) if pd.notna(x)]

    filtered_data = orgX.reset_index(drop=True)
    time_columns = ['timestamp_unix', 'timestamp_iso', 'timestamp']  # , 'timestamp_israel']
    cols_to_remove = time_columns + ['missing_value_reason'] + ['prv_rmssd_ms',
                                                                'respiratory_rate_brpm',
                                                                # 'pulse_rate_bpm'
                                                                ] + ['eventType']

    cols_to_remove = cols_to_remove + (['severity'] if 'severity' in orgX.keys() else [])

    cols = [c for c in filtered_data.keys() if c not in cols_to_remove]

    X = filtered_data[cols]

    # Handle categorical columns with one-hot encoding
    def create_encoding(df, col, categories):
        if categories:
            df[col] = df[col].astype(
                pd.CategoricalDtype(categories=categories)
            )
        df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=True)
        return df

    # X = create_encoding(X, 'eventType', eventType_classes)
    # X = create_encoding(X, 'activity_intensity', activity_intensity_classes)
    # X = create_encoding(X, 'body_position_left', body_position_left_classes)
    # X = create_encoding(X, 'body_position_right', body_position_right_classes)

    # print(f"Data Shape: {X.shape}")
    return X


def prep_data(positive_data, negative_data, multiclassification=False, participent_in_test=None, split_by_time=None,
              standard_scale=False, num_weeks=None):
    if multiclassification:
        positive_data['classification'] = positive_data['severity']

        mapping = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
        positive_data["classification"] = positive_data["severity"].map(mapping)
        negative_data['classification'] = 0
        negative_data['severity'] = 0
    else:
        positive_data['classification'] = 1
        negative_data['classification'] = 0

        negative_data['eventType'] = 'None'

    orgX = pd.concat([positive_data, negative_data])
    orgX = orgX.sort_values('timestamp_israel')

    class_counts = orgX['classification'].value_counts()
    print(class_counts / len(orgX))

    # X = prep_onehotencoded_columns(orgX)
    X = orgX
    X["hour"] = X["timestamp_israel"].dt.hour
    X["dow"] = X["timestamp_israel"].dt.dayofweek
    X["dom"] = X["timestamp_israel"].dt.day
    if participent_in_test:
        X_test = X[X['participant_full_id'].str.contains(participent_in_test, na=False)]
        X_train = X[~X['participant_full_id'].str.contains(participent_in_test, na=False)]

        # X_train, _ = self.downsample_data(X_train, precentage_of_1=30)
        if num_weeks:
            # Find the first timestamp in the DataFrame
            start_date = X_test['timestamp_israel'].min()
            # Define the end of the first week
            end_date = start_date + pd.Timedelta(weeks=i)
            X_week_data = X_test[(X_test['timestamp_israel'] >= start_date) & (X_test['timestamp_israel'] < end_date)]
            X_test.drop(X_week_data.index)
            X_train = X_week_data

        y_train = X_train['classification']
        y_test = X_test['classification']
    else:
        X_train, X_test = split_train_test_by_day(X)
        X_train, X_val = split_train_test_by_day(X_train)

        y_train = X_train['classification']
        y_test = X_test['classification']
        y_val = X_val['classification']

    X_train = X_train.drop(['classification', 'eventType', 'participant_full_id', 'timestamp_israel'], axis=1)
    X_val = X_val.drop(['classification', 'eventType', 'participant_full_id', 'timestamp_israel'], axis=1)
    X_test = X_test.drop(['classification', 'eventType', 'participant_full_id', 'timestamp_israel'], axis=1)

    if standard_scale:
        scaler = StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        return X_train, X_val, X_test, y_train, y_val, y_test

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == '__main__':
    patients_dict = {
        'TRAIL001': 'TRAIL001-3YK3L151K2',
        'TRAIL002': 'TRAIL002-3YK3J1514F',
        'TRAIL003': 'TRAIL003-3YK3K153QJ',
        'TRAIL004': 'TRAIL004-3YK3J151CV',
        'TRAIL005': 'TRAIL005-3YK3L151DR',
        'TRAIL008': 'TRAIL008-3YK3J1514F',
        'TRAIL009': 'TRAIL009-3YKC51P1YL',
        'TRAIL010': 'TRAIL10-3YKC51P2H3',
        'TRAIL011': 'TRAIL011-3YK3L151DR',
        'TRAIL012': 'TRAIL012-3YK3L151K2',
        'TRAIL013': 'TRAIL013-3YK3J1514F',
    }

    trail_dates = {
        'TRAIL003': {'start_date': '30.3.2025 16:04', 'end_date': '29.4.2025 17:40'},
        'TRAIL002': {'start_date': '9.4.2025 14:28', 'end_date': '8.5.2025 19:30'},
        'TRAIL001': {'start_date': '24.4.2025 11:40', 'end_date': '23.5.2025 10:21'},
        'TRAIL004': {'start_date': '27.4.2025 15:00', 'end_date': '26.5.2025 14:25'},
        'TRAIL005': {'start_date': '14.5.2025 11:56', 'end_date': '17.6.2025 00:00'},
        'TRAIL008': {'start_date': '10.7.2025 14:50', 'end_date': '10.8.2025 00:00'},
        'TRAIL009': {'start_date': '31.7.2025 13:15', 'end_date': '29.8.2025 10:00'},
        'TRAIL010': {'start_date': '3.8.2025 10:48', 'end_date': '1.9.2025 12:00'},
        'TRAIL011': {'start_date': '14.8.2025 14:00', 'end_date': '21.9.2025 13:00'},
        'TRAIL012': {'start_date': '28.8.2025 14:00', 'end_date': '25.9.2025 10:00'},
        'TRAIL013': {'start_date': '31.8.2025 12:00', 'end_date': '25.9.2025 12:00'},
    }
    fmt = "%d.%m.%Y %H:%M"
    trail_dates_ts = {
        k: {
            'start_date': datetime.strptime(v['start_date'], fmt),
            'end_date': datetime.strptime(v['end_date'], fmt),
        }
        for k, v in trail_dates.items()
    }

    time = '15min'
    window_minutes = 60 * 3
    step_minutes = 60 * 3
    normalize = False
    standard_scaling = False
    multiclassification = False

    tags_path = r'../data\embrace_plus\participants_extra_data\valid_tags'
    data_path = r'C:\Users\GONY\Desktop\Booggii\data'
    chunked_data_path = fr"C:\Users\GONY\Desktop\Booggii\processed_data\old_classification_tod_features{window_minutes}min_{step_minutes}step{'_normalized_' if normalize else ''}"

    if os.path.exists(chunked_data_path):
        # if False:
        print('loading existing pickle files:')
        positive_data = pd.read_pickle(
            chunked_data_path + rf"\positive_data_{'normalized_' if normalize else ''}{window_minutes}min_{step_minutes}step.pkl")
        negative_data = pd.read_pickle(
            chunked_data_path + rf"\negative_data_{'normalized_' if normalize else ''}{window_minutes}min_{step_minutes}step.pkl")
    else:
        os.makedirs(chunked_data_path, exist_ok=True)
        print('creating data')

        positive_data, negative_data = prepare_biomarkers_data(patients_dict, tags_path, data_path, time,
                                                               trail_dates_ts)
        positive_data['classification'] = 1
        negative_data['classification'] = 0
        data = pd.concat([positive_data, negative_data])
        data = data.sort_values(by='timestamp_israel')
        data = create_chunked_data(data, patients_dict, window_minutes=window_minutes,
                                            step_minutes=step_minutes, enable_tod=True)
        positive_data = data[data['classification']==1]
        negative_data = data[data['classification'] == 0]
        # positive_data = create_chunked_data(positive_data, patients_dict, window_minutes=window_minutes,
        #                                     step_minutes=step_minutes, enable_tod=True)
        positive_data.to_pickle(
            chunked_data_path + rf'\positive_data_{window_minutes}min_{step_minutes}step.pkl')
        negative_data.to_pickle(
            chunked_data_path + rf'\negative_data_{window_minutes}min_{step_minutes}step.pkl')

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M")
    time_str = "new__code_XGBoost_withpulse" + ('multiclassification_' if multiclassification else '') + time_str
    output_dir = os.path.join(r'C:\Users\GONY\Desktop\Booggii\results\xgboost_output', 'results', time_str)
    os.makedirs(output_dir, exist_ok=True)

    for patient in patients_dict.keys():
        print(f'XGBoost training on patient {patient}')
        temp_path = os.path.join(output_dir, f'eval_{patient}')
        pos_data = positive_data[positive_data['participant_full_id'].str.contains(patient, na=False)]
        neg_data = negative_data[negative_data['participant_full_id'].str.contains(patient, na=False)]
        os.makedirs(temp_path, exist_ok=True)
        X_train, X_val, X_test, y_train, y_val, y_test = prep_data(pos_data, neg_data)

        print(
            f'positive events in train set: {(sum(y_train) / len(y_train)) * 100}%\npositive events in val set: {(sum(y_val)/len(y_val))*100}%\npositive events in test set: {(sum(y_test)/len(y_test))*100}%')

        xgboost = xgboost_model(X_train, X_val, X_test, y_train, y_val, y_test, output_dir=temp_path,
                                multiclassification=False,
                                n_estimators=100, learning_rate=0.01, subsample=0.8, max_depth=10
                                )
        xgboost.run_model(grid_search=True)
