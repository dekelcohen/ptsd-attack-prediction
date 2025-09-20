import numpy as np
import pandas as pd
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, PrecisionRecallDisplay, precision_recall_curve

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_preparation import prepare_biomarkers_data


class xgboost_model():
    def __init__(self, X_train, X_test, y_train, y_test, multiclassification, output_dir):
        self.multiclassification = multiclassification
        self.output_dir = output_dir
        num_classes = set(pd.concat([y_train, y_test]))
        y_train_count = y_train.value_counts()
        scale_pos_weight = y_train_count[0] / y_train_count[1]
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

        if multiclassification:
            self.model = xgb.XGBClassifier(
                random_state=42,
                objective="multi:softprob ",  # or "multi:softmax"
                num_class=num_classes,
                eval_metric="mlogloss"
            )
        else:
            self.model = xgb.XGBClassifier(
                random_state=42,
                objective="binary:logistic",
                use_label_encoder=False,
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight
            )

    def run_model(self):
        self.model.fit(self.X_train, self.y_train)

        if self.multiclassification:
            y_pred_probs = self.model.predict_proba(self.X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)  # best class for each sample
        else:
            y_category_pred = self.model.predict(self.X_test)
            print("\n--- Initial Classification Report (with custom threshold) ---")
            print(classification_report(self.y_test, y_category_pred, target_names=['No Stress', 'Stress']))

            # Get the probability scores for the positive class (Stress)
            y_pred = self.model.predict_proba(self.X_test)[:, 1]

            # --- Plot the Precision-Recall Curve ---
            display = PrecisionRecallDisplay.from_predictions(self.y_test, y_pred, name="XGBoost")
            plt.title("Precision-Recall Curve")
            plt.savefig(os.path.join(self.output_dir, "precision_recall_curve.png"), dpi=300, bbox_inches="tight")

            # 1. Get precision, recall, and thresholds
            precision, recall, thresholds = precision_recall_curve(self.y_test, y_pred)

            # 2. Find the threshold that maximizes the F1-score
            # F1 = 2 * (precision * recall) / (precision + recall)
            # We add a small number (1e-9) to avoid division by zero
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)

            # The last precision and recall values are 1. and 0. respectively and do not have a corresponding threshold.
            # So we slice the F1 scores to match the number of thresholds.
            best_threshold = thresholds[np.argmax(f1_scores[:-1])]

            print(f"Best Threshold based on F1-Score: {best_threshold:.4f}")

            # 3. Apply the new threshold to your probability scores
            new_predictions = (y_pred >= best_threshold).astype(int)

            # 4. Compare the new classification report with the old one
            print("\n--- New Threshold Classification Report (with custom threshold) ---")
            report = classification_report(self.y_test, new_predictions, target_names=['No Stress', 'Stress'])
            with open(f"{self.output_dir}/classification_report.txt", "w") as f:
                f.write(report)
            print(report)

            cm = confusion_matrix(self.y_test, new_predictions)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion Matrix")
            plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
            plt.close()
            print(self.output_dir)

    def plot_important_features(self):
        plt.figure(figsize=(10, 8))  # adjust width/height as needed
        xgb.plot_importance(self.model)

        plt.tight_layout()  # makes sure labels and titles fit into the figure
        plt.savefig(os.path.join(self.output_dir, "feature_importance.png"), dpi=300)
        plt.close()


def prep_onehotencoded_columns(orgX, data):
    # activity_classes = [x for x in list(set(orgX['activity_class'])) if pd.notna(x)]
    # activity_intensity_classes = [x for x in list(set(orgX['activity_intensity'])) if pd.notna(x)]
    # body_position_left_classes = [x for x in list(set(orgX['body_position_left'])) if pd.notna(x)]
    # body_position_right_classes = [x for x in list(set(orgX['body_position_right'])) if pd.notna(x)]

    filtered_data = data.reset_index(drop=True)
    filtered_data["hour"] = filtered_data["timestamp_israel"].dt.hour
    filtered_data["dow"] = filtered_data["timestamp_israel"].dt.dayofweek
    filtered_data["dom"] = filtered_data["timestamp_israel"].dt.day
    time_columns = ['timestamp_unix', 'timestamp_iso', 'timestamp', 'timestamp_israel']
    cols_to_remove = time_columns + ['missing_value_reason'] + ['prv_rmssd_ms',
                                                                'respiratory_rate_brpm',
                                                                # 'pulse_rate_bpm'
                                                                ] + ['eventType']

    cols_to_remove = cols_to_remove + (['severity'] if 'severity' in data.keys() else [])

    cols = [c for c in filtered_data.keys() if c not in cols_to_remove]

    X = filtered_data[cols]

    # X = X.dropna(subset=["pulse_rate_bpm"])

    # Handle categorical columns with one-hot encoding
    def create_encoding(df, col, categories):
        if categories:
            df[col] = df[col].astype(
                pd.CategoricalDtype(categories=categories)
            )
        df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=True)
        return df

    # X = create_encoding(X, 'activity_class', activity_classes)
    # X = create_encoding(X, 'activity_intensity', activity_intensity_classes)
    # X = create_encoding(X, 'body_position_left', body_position_left_classes)
    # X = create_encoding(X, 'body_position_right', body_position_right_classes)

    print(f"Data Shape: {X.shape}")
    print(f"Columns: {list(X.columns)}")
    return X


def downsample_data(orgX, precentage_of_1=50, multiclassification=False):
    if multiclassification:
        class_counts = orgX['classification'].value_counts()
        print(f"Original class distribution:\n{class_counts}")

        # Calculate median size
        median_size = int(class_counts.median())
        print(f"\nMedian class size: {median_size}")

        downsampled_dfs = []
        leftover_dfs = []

        for class_label in class_counts.index:
            class_data = orgX[orgX['classification'] == class_label]

            if len(class_data) > median_size:
                # Downsample if class is larger than median
                downsampled_class = class_data.sample(n=median_size, random_state=42)
                leftover_class = class_data.drop(downsampled_class.index)
            else:
                # Keep all samples if class is smaller than or equal to median
                downsampled_class = class_data
                leftover_class = pd.DataFrame(columns=class_data.columns)  # empty

            downsampled_dfs.append(downsampled_class)
            leftover_dfs.append(leftover_class)

        df_balanced = pd.concat(downsampled_dfs, ignore_index=True)
        df_remaining = pd.concat(leftover_dfs, ignore_index=True)

    else:
        df_majority = orgX[orgX['classification'] == 0]
        df_minority = orgX[orgX['classification'] == 1]
        # n_total = min(len(orgX), int(len(df_minority) * (100 / precentage_of_1)))
        # n_majority = int(n_total * ((100 - precentage_of_1) / 100))
        # n_minority = int(n_total * (precentage_of_1 / 100))

        df_majority_down = resample(
            df_majority,
            replace=False,
            # n_samples=min(n_majority, len(df_majority)),  # safeguard
            n_samples=len(df_minority),
            random_state=42
        )
        df_balanced = pd.concat([df_majority_down, df_minority])

        # Remaining rows = majority rows not sampled
        df_remaining = df_majority.drop(df_majority_down.index)

    # shuffle balanced set
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced


def prep_data(positive_data, negative_data, multiclassification=False, participent_in_test=None, split_by_time=False):
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
    print(class_counts)
    print(class_counts / len(orgX))

    X = prep_onehotencoded_columns(orgX, orgX)

    if participent_in_test:
        X_test = X[X['participant_full_id'].str.contains(participent_in_test, na=False)]
        X_train = X[~X['participant_full_id'].str.contains(participent_in_test, na=False)]
        # X_train, _ = self.downsample_data(X_train, precentage_of_1=30)

        y_train = X_train['classification']
        y_test = X_test['classification']


    else:
        # X = downsample_data(X)
        y = X['classification']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    X_train = X_train.drop(['classification'], axis=1)
    X_train = X_train.drop(['participant_full_id'], axis=1)
    X_test = X_test.drop(['classification'], axis=1)
    X_test = X_test.drop(['participant_full_id'], axis=1)

    return X_train, X_test, y_train, y_test


def plot_prediction_time_graphs(df, output_dir):
    for day in df['timestamp_israel'].dt.date.unique():
        for_plot = df[df['timestamp_israel'].dt.date == day].copy()
        for_plot["timestamp_israel"] = pd.to_datetime(for_plot["timestamp_israel"])

        df_pivot = for_plot.pivot_table(index="timestamp_israel", values=['true', 'pred'], aggfunc='mean')
        if df_pivot.empty or df_pivot.size == 0:
            print("[heatmap] nothing to plot: empty after cleaning")
            continue

        df_pivot = df_pivot.sort_index()
        sns.heatmap(df_pivot.T, cmap="coolwarm", cbar=False)
        plt.tight_layout()
        plt.savefig(output_dir + f'/for_dom_{day}.png')
        plt.close()



if __name__ == '__main__':
    patients_dict = {'TRAIL001': 'TRAIL001-3YK3L151K2',
                     'TRAIL002': 'TRAIL002-3YK3J1514F',
                     'TRAIL003': 'TRAIL003-3YK3K153QJ',
                     'TRAIL004': 'TRAIL004-3YK3J151CV',
                     'TRAIL005': 'TRAIL005-3YK3L151DR'}

    tags_path = r'../data\embrace_plus\participants_extra_data\valid_tags'
    data_path = r'C:\Users\GONY\Desktop\Booggii\data'
    time = '15min'
    positive_data, negative_data = prepare_biomarkers_data(patients_dict, tags_path, data_path, time)

    multiclassification = False
    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M")
    time_str = "new__code_XGBoost_withpulse" + ('multiclassification_' if multiclassification else '') + time_str
    output_dir = os.path.join(r'C:\Users\GONY\Desktop\Booggii\results\xgboost_output', 'results', time_str)
    os.makedirs(output_dir, exist_ok=True)

    #### ALL DATA
    # X_train, X_test, y_train, y_test = prep_data(positive_data, negative_data, multiclassification=multiclassification)
    # xgboost = xgboost_model(X_train, X_test, y_train, y_test, multiclassification=multiclassification,
    #                         output_dir=output_dir)
    #
    # xgboost.run_model()

    #### TRAINED SEPERATLY FOR EACH PARTICIPANT
    positive_data['id'] = np.arange(1, 1 + len(positive_data))
    negative_data['id'] = np.arange(1 + len(positive_data), 1 + len(positive_data) + len(negative_data))
    for patient in patients_dict.keys():
        print(f'XGBoost trained only for patient {patient}')
        pos_data = positive_data[positive_data['participant_full_id'].str.contains(patient, na=False)]
        neg_data = negative_data[negative_data['participant_full_id'].str.contains(patient, na=False)]
        temp_path = os.path.join(output_dir, f'trained_on_{patient}')
        os.makedirs(temp_path, exist_ok=True)
        X_train, X_test, y_train, y_test = prep_data(pos_data, neg_data)
        xgboost = xgboost_model(X_train.drop(columns=['id']), X_test.drop(columns=['id']), y_train, y_test, output_dir=temp_path, multiclassification=False)

        xgboost.run_model()

        y_pred = xgboost.model.predict_proba(X_test.drop(columns=['id']))[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
        best_threshold = thresholds[np.argmax(f1_scores[:-1])]
        new_predictions = (y_pred >= best_threshold).astype(int)
        X_test['pred'] = new_predictions
        X_test['true'] = y_test

        # Build a single {id -> timestamp} map from df1 & df2
        ts_lookup = (
            pd.concat([
                pos_data[['id', 'timestamp_israel']].dropna(subset=['timestamp_israel']),
                neg_data[['id', 'timestamp_israel']].dropna(subset=['timestamp_israel'])
            ])
            .drop_duplicates('id', keep='last')  # in case of overlap
            .set_index('id')['timestamp_israel']
        )

        # Assign (or fill) in df3
        if 'timestamp_israel' in X_test:
            X_test['timestamp_israel'] = X_test['timestamp_israel'].fillna(X_test['id'].map(ts_lookup))
        else:
            X_test['timestamp_israel'] = X_test['id'].map(ts_lookup)
        temp_path = os.path.join(temp_path, f'prediction_time_graphs')
        os.makedirs(temp_path, exist_ok=True)
        plot_prediction_time_graphs(X_test, temp_path)

    #### CROSS VALIDATION ON EACH PARTICIPANT
    # positive_data['id'] = np.arange(1, 1 + len(positive_data))
    # negative_data['id'] = np.arange(1 + len(positive_data), 1 + len(positive_data) + len(negative_data))
    # for patient in patients_dict.keys():
    #     print(f'XGBoost tested on patient {patient}')
    #     temp_path = os.path.join(output_dir, f'tested_on_{patient}')
    #     os.makedirs(temp_path, exist_ok=True)
    #     X_train, X_test, y_train, y_test = prep_data(positive_data, negative_data, participent_in_test=patient)
    #
    #     xgboost = xgboost_model(X_train.drop(columns=['id']), X_test.drop(columns=['id']), y_train, y_test, output_dir=temp_path, multiclassification=False)
    #
    #     xgboost.run_model()
    #
    #     y_pred = xgboost.model.predict_proba(X_test.drop(columns=['id']))[:, 1]
    #     precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    #     f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    #     best_threshold = thresholds[np.argmax(f1_scores[:-1])]
    #     new_predictions = (y_pred >= best_threshold).astype(int)
    #     X_test['pred'] = new_predictions
    #     X_test['true'] = y_test
    #
    #     # Build a single {id -> timestamp} map from df1 & df2
    #     ts_lookup = (
    #         pd.concat([
    #             positive_data[['id', 'timestamp_israel']].dropna(subset=['timestamp_israel']),
    #             negative_data[['id', 'timestamp_israel']].dropna(subset=['timestamp_israel'])
    #         ])
    #         .drop_duplicates('id', keep='last')  # in case of overlap
    #         .set_index('id')['timestamp_israel']
    #     )
    #
    #     # Assign (or fill) in df3
    #     if 'timestamp_israel' in X_test:
    #         X_test['timestamp_israel'] = X_test['timestamp_israel'].fillna(X_test['id'].map(ts_lookup))
    #     else:
    #         X_test['timestamp_israel'] = X_test['id'].map(ts_lookup)
    #     temp_path = os.path.join(temp_path, f'prediction_time_graphs')
    #     os.makedirs(temp_path, exist_ok=True)
    #     plot_prediction_time_graphs(X_test, temp_path)