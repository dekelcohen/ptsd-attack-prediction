import os
from enum import Enum
from pathlib import Path

import pandas as pd
import pytz
import neurokit2 as nk
from bokeh.layouts import column, gridplot
from bokeh.models import DatetimeTickFormatter
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from matplotlib import pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np

import avro_utils


from biomarkers import Biomarker, biomarker_value_names, biomarker_colors


def main():

    # user_id = "TRAIL005"
    # trial_starting_date = "2025-05-14 11:56:00"
    # user_id = "TRAIL009"
    # trial_starting_date = "2025-07-31 14:50:00"
    # user_id = "TRAIL017"
    # trial_starting_date = "2025-12-07 08:30:00"
    # user_id = "TRAIL004"
    # trial_starting_date = "2025-04-27 15:00:00"
    # user_id = "TRAIL004"
    # trial_starting_date = "2025-04-27 15:00:00"
    # user_id = "TRAIL008"
    # trial_starting_date = "2025-07-10 14:50:00"
    user_id = "TRAIL10"
    trial_starting_date = "2025-08-03 10:48:00"



    jerusalem_tz = pytz.timezone('Asia/Jerusalem')
    data_root_dir = Path("data/embrace_plus/")

    biomarker_names = [Biomarker.Pr, Biomarker.AccStd, Biomarker.Temp,
                       Biomarker.EdaPhasic, Biomarker.EdaTonic, Biomarker.Prv, Biomarker.Met, Biomarker.RR]
    biomarker_dfs, raw_tags_df, modified_tags_df = prepare_data_and_tags(biomarker_names, data_root_dir, jerusalem_tz, trial_starting_date,
                                                       user_id, override=False)

    anomaly_df = detect_anomalies(biomarker_dfs)

    # visualize_data(biomarker_dfs, raw_tags_df, modified_tags_df, anomaly_events=anomaly_df, split=True)
    normalize = True
    visualize_events(biomarker_dfs, raw_tags_df, modified_tags_df, anomaly_events=anomaly_df, time_delta=pd.Timedelta(hours=6), normalize=normalize)
    # visualize_statistics(biomarker_dfs)


def visualize_statistics(biomarker_dfs):
    merged_bio = pd.concat([x.set_index(['datetime']) for x in list(biomarker_dfs.values())], axis=1)
    merged_bio_desc = merged_bio.describe(percentiles=[.05, .25, .5, .75, .95])
    fig1 = plt.figure(1)
    fig2 = plt.figure(2)
    ax1 = fig1.subplots()
    ax2 = fig2.subplots()
    ax1.axis('off')
    ax2.axis('off')
    merged_bio_desc_table = pd.plotting.table(ax1, merged_bio_desc, loc='center',
                                              cellLoc='left')  # , colWidths=list([.2, .2]))
    merged_bio_desc_table.auto_set_font_size(False)
    merged_bio_desc_table.set_fontsize(10)
    merged_bio_corr = merged_bio.corr(method='spearman')
    merged_bio_corr.style.background_gradient(cmap='coolwarm')
    merged_bio_corr_table = pd.plotting.table(ax2, merged_bio_corr, loc='center', cellLoc='right')
    merged_bio_corr_table.auto_set_font_size(False)
    merged_bio_corr_table.set_fontsize(10)
    for biomarker_name, biomarker_df in biomarker_dfs.items():
        biomarker_df.plot.hist(bins=20, alpha=0.5)
    plt.show()


def visualize_events(biomarker_dfs, raw_tags_df, modified_tags_df, anomaly_events=None, time_delta=pd.Timedelta(hours=1), normalize=False):
    event_plots = []
    
    if not raw_tags_df.empty:
        for i in range(len(raw_tags_df)):
            if 'datetime' not in raw_tags_df.columns:
                continue
                
            raw_event_time = raw_tags_df.iloc[i]['datetime']
            
            modified_event_time = None
            if not modified_tags_df.empty and i < len(modified_tags_df) and 'datetime' in modified_tags_df.columns:
                 modified_event_time = modified_tags_df.iloc[i]['datetime']

            filtered_biomarker_dfs = {}
            # Define window around the RAW event time
            start_time = raw_event_time - time_delta
            end_time = raw_event_time + time_delta

            for biomarker_name, biomarker_df in biomarker_dfs.items():
                window_data = biomarker_df[
                    (biomarker_df['datetime'] >= start_time) & (biomarker_df['datetime'] <= end_time)]

                if not window_data.empty:
                    filtered_biomarker_dfs[biomarker_name] = window_data
                elif normalize:
                     # For normalized plots, we might want to ensure we don't break if empty
                     pass
                else:
                    pass
            
            # Filter anomalies for this window
            window_anomalies = None
            if anomaly_events is not None and not anomaly_events.empty:
                 window_anomalies = anomaly_events[
                    (anomaly_events['datetime'] >= start_time) & (anomaly_events['datetime'] <= end_time)
                 ]
                 
            event_plots.append(visualize_data(filtered_biomarker_dfs, raw_event_time, modified_event_time, anomaly_events=window_anomalies, split=False, normalize=normalize))
    
    show(gridplot(event_plots, ncols=1, sizing_mode="stretch_width"))


from sklearn.ensemble import IsolationForest

def detect_anomalies(biomarker_dfs):
    # Combine dataframes to align timestamps
    # We will prioritize EDA and Phasic/Tonic if available, or just common index
    dfs_to_merge = []
    for name, df in biomarker_dfs.items():
        if not df.empty:
             # Rename value column to biomarker name to avoid collisions
             temp_df = df.set_index('datetime').rename(columns={biomarker_value_names[name]: name.value})
             # Resample to 1min or kept original? data_visualization seems to use original or varying freq.
             # We should probably resample to a common frequency for alignment, e.g., 1s or 1min.
             # Let's try to match on existing timestamps (inner join) or nearest.
             # Given the data is likely high frequency for some (EDA 4Hz) and low for others (HR 1Hz),
             # and processed data in data_prep creates per-minute or high freq.
             
             # Let's resample to 1 minute for anomaly detection to be robust and fast
             temp_df = temp_df.resample('1min').mean()
             dfs_to_merge.append(temp_df)
    
    if not dfs_to_merge:
        return pd.DataFrame()

    combined_df = pd.concat(dfs_to_merge, axis=1).dropna()
    
    if combined_df.empty:
        return pd.DataFrame()

    # Train Isolation Forest
    # clf = IsolationForest(n_estimators=300, max_samples=256, contamination=0.05, random_state=42)
    # Fit and predict
    # 1 for inliers, -1 for outliers
    # preds = clf.fit_predict(combined_df)
    # anomalies = combined_df[preds == -1]

    tuned_results = tune_isolation_forest(combined_df)
    anomalies = tuned_results[tuned_results['predicted_stress'] == True]
    return anomalies.reset_index()[['datetime']]


def tune_isolation_forest(features_df):
    """
     tunes Isolation Forest by analyzing the distribution of anomaly scores
     rather than guessing a contamination percentage.
    """

    # 1. Feature Scaling
    # Isolation Forest is somewhat robust to scale, but scaling helps
    # when combining units like ms (HRV) and count (EDA).
    scaler = StandardScaler()
    X = scaler.fit_transform(features_df)

    # 2. Train Model (Optimized for Stability)
    # We use n_estimators=300 for stability and max_samples=256 to prevent swamping.
    iso_forest = IsolationForest(
    n_estimators = 300,
    max_samples = 256,
    contamination = 0.5,
    random_state = 42,
    n_jobs = -1

    )
    iso_forest.fit(X)

    # 3. Get Raw Anomaly Scores (The "Tuning" Step)
    # The decision_function returns negative values for outliers, positive for inliers.
    # We invert this so higher score = more anomalous (more stressed).
    raw_scores = -1 * iso_forest.decision_function(X)
    features_df['stress_score'] = raw_scores

    # 4. Visualization for Threshold Tuning
    plt.figure(figsize=(10, 6))
    sns.histplot(raw_scores, bins=50, kde=True, color='blue', alpha=0.6)

    # Calculate Statistical Thresholds
    # Option A: 3 Sigma (very conservative, catches only extreme panic)
    thresh_3std = np.mean(raw_scores) + 3 * np.std(raw_scores)

    # Option B: IQR Rule (robust to outliers, recommended for physiological data)
    Q1 = np.percentile(raw_scores, 25)
    Q3 = np.percentile(raw_scores, 75)
    IQR = Q3 - Q1
    thresh_iqr = Q3 + 1.5 * IQR

    plt.axvline(thresh_3std, color='red', linestyle='--', label=f'3-Sigma Threshold ({thresh_3std:.2f})')
    plt.axvline(thresh_iqr, color='green', linestyle='--', label=f'IQR Threshold ({thresh_iqr:.2f})')

    plt.title('Distribution of Stress Scores: Where is the Cutoff?')
    plt.xlabel('Anomaly Score (Higher = More Stressed)')
    plt.legend()
    # plt.show()

    print(f"Recommended Threshold (IQR Method): {thresh_iqr:.3f}")

    # Apply Threshold
    features_df['predicted_stress'] = features_df['stress_score'] > thresh_iqr
    return features_df


from sklearn.preprocessing import StandardScaler, MinMaxScaler
from bokeh.models import Range1d, LinearAxis

def visualize_data(biomarker_dfs, raw_events, modified_events, anomaly_events=None, split=True, normalize=False):
    # Single plot with multiple Y-axes
    p = figure(sizing_mode="stretch_width", height=400, x_axis_type='datetime', background_fill_color="WhiteSmoke", title="Event Analysis")
    p.xaxis.formatter = DatetimeTickFormatter(days="%d/%m", hours="%H", minutes="%H:%M")
    
    first_biomarker = True
    
    for name, df in biomarker_dfs.items():
        if df.empty: continue
        
        y_col = biomarker_value_names[name]
        
        # Calculate range for this biomarker
        y_min = df[y_col].min()
        y_max = df[y_col].max()
        
        # Add buffer
        range_span = y_max - y_min
        if range_span == 0: range_span = 1
        y_start = y_min - 0.1 * range_span
        y_end = y_max + 0.1 * range_span
        
        if first_biomarker:
            # Main Axis
            p.y_range = Range1d(start=y_start, end=y_end)
            p.yaxis.axis_label = name.value
            
            p.line(
                x=df["datetime"].dt.tz_localize(None),
                y=df[y_col],
                legend_label=name.value,
                color=biomarker_colors[name],
                line_width=2
            )
            first_biomarker = False
        else:
            # Extra Axis
            p.extra_y_ranges[name.value] = Range1d(start=y_start, end=y_end)
            
            # Add the new axis
            ax = LinearAxis(y_range_name=name.value, axis_label=name.value)
            p.add_layout(ax, 'right')
            
            p.line(
                x=df["datetime"].dt.tz_localize(None),
                y=df[y_col],
                legend_label=name.value,
                color=biomarker_colors[name],
                y_range_name=name.value,
                line_width=2
            )

    # Prepare event data once
    localized_raw_events = []
    if isinstance(raw_events, DataFrame) and not raw_events.empty:
         localized_raw_events = raw_events["datetime"].dt.tz_localize(None)
    elif isinstance(raw_events, pd.Timestamp):
         localized_raw_events = [raw_events.tz_localize(None)]
         
    localized_mod_events = []
    if isinstance(modified_events, DataFrame) and not modified_events.empty:
         localized_mod_events = modified_events["datetime"].dt.tz_localize(None)
    elif isinstance(modified_events, pd.Timestamp):
         localized_mod_events = [modified_events.tz_localize(None)]

    localized_anomalies = []
    if anomaly_events is not None and not anomaly_events.empty:
        localized_anomalies = anomaly_events["datetime"].dt.tz_localize(None)

    # Add Vspans to the plot
    if len(localized_raw_events) > 0:
        p.vspan(x=localized_raw_events, line_color="red", legend_label="raw events",
                alpha=1.0, line_width=2)
    
    if len(localized_mod_events) > 0:
        p.vspan(x=localized_mod_events, line_color="green", legend_label="modified events",
                alpha=1.0, line_width=2)
    
    if len(localized_anomalies) > 0:
        p.vspan(x=localized_anomalies, line_color="orange", line_width=2, legend_label="anomalies",
                alpha=0.5)

    # Tools
    tooltips = [('datetime', '@x{%Y-%m-%d %H:%M:%S}'), ('value', '@y')]
    p.add_tools(HoverTool(tooltips=tooltips, formatters={'@x': 'datetime'}))
    
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"

    return p


def prepare_data_and_tags(biomarker_names, data_root_dir, jerusalem_tz, trial_starting_date,
                          user_id, override=False):
    biomarker_dfs = prepare_data(biomarker_names, data_root_dir, jerusalem_tz, override, trial_starting_date, user_id)

    modified_tags_df = pd.DataFrame()
    mod_path_str = "data/embrace_plus/participants_extra_data/valid_tags/auto_modified_tags/" + user_id + "_valid_tags_modified.csv"
    if os.path.exists(mod_path_str):
        modified_tags_df = pd.read_csv(mod_path_str, sep=',')

    raw_tags_df = pd.DataFrame()

    if not modified_tags_df.empty:
        # Use timestamp from modified tags as raw tags
        if "timestamp" in modified_tags_df.columns:
            # Create raw_tags_df from modified_tags_df['timestamp']
            raw_tags_df = modified_tags_df.copy()
            raw_tags_df["timestamp"] = raw_tags_df["timestamp"].astype(str).str.replace(" IDT", "").str.replace(" IST", "")
            raw_tags_df["datetime"] = pd.to_datetime(raw_tags_df['timestamp']).dt.tz_localize(
                 jerusalem_tz, ambiguous='NaT', nonexistent='NaT')
        
        # Prepare modified tags (new_timestamp)
        if "new_timestamp" in modified_tags_df.columns:
             modified_tags_df["timestamp"] = modified_tags_df["new_timestamp"]
             modified_tags_df["timestamp"] = modified_tags_df["timestamp"].astype(str).str.replace(" IDT", "").str.replace(" IST", "")
             modified_tags_df["datetime"] = pd.to_datetime(modified_tags_df['timestamp']).dt.tz_localize(
                 jerusalem_tz, ambiguous='NaT', nonexistent='NaT')

    else:
        # Fallback to reading raw tags file if modified doesn't exist
        raw_path_str = "data/embrace_plus/participants_extra_data/valid_tags/raw_tags/" + user_id + "_valid_tags_raw.csv"
        if os.path.exists(raw_path_str):
             raw_tags_df = pd.read_csv(raw_path_str, sep=',')
             if not raw_tags_df.empty:
                if "timestamp" in raw_tags_df.columns:
                     raw_tags_df["timestamp"] = raw_tags_df["timestamp"].astype(str).str.replace(" IDT", "").str.replace(" IST", "")
                     raw_tags_df["datetime"] = pd.to_datetime(raw_tags_df['timestamp']).dt.tz_localize(
                    jerusalem_tz, ambiguous='NaT', nonexistent='NaT')

    return biomarker_dfs, raw_tags_df, modified_tags_df


def prepare_data(biomarker_names, data_root_dir, jerusalem_tz, override, trial_starting_date, user_id):
    participant_data_dir = data_root_dir.joinpath("participant_data")
    participant_processed_data_dir = data_root_dir.joinpath("participant_processed_data")
    cache_file = participant_processed_data_dir.joinpath(f"{user_id}_biomarker_dfs.pkl")
    if not override and cache_file.exists():
        print(f"Loading processed data from cache: {cache_file}")
        biomarker_dfs = pd.read_pickle(cache_file)
    else:
        trial_starting_datetime = pd.to_datetime(trial_starting_date).tz_localize('Asia/Jerusalem')
        biomarker_dfs = {biomarker_name: DataFrame() for biomarker_name in biomarker_names}
        all_tags_df = DataFrame()
        all_hr_df = DataFrame()
        all_eda_df = DataFrame()
        all_temp_df = DataFrame()
        for date_dir in os.listdir(participant_data_dir):
            date_path = participant_data_dir.joinpath(date_dir)
            # if date_dir != "2025-04-21":
            #     continue
            for user_dir in os.listdir(date_path):
                if user_dir.startswith(user_id):
                    print(f"processing user {user_dir} on date {date_dir}")
                    user_processed_data_dir = participant_processed_data_dir.joinpath(user_dir)
                    if not user_processed_data_dir.exists():
                        user_processed_data_dir.mkdir()

                    user_path = date_path.joinpath(user_dir)
                    biomarkers_path = user_path.joinpath("digital_biomarkers/aggregated_per_minute/")
                    for biomarker_file in os.listdir(biomarkers_path):
                        for biomarker_name in biomarker_names:
                            if biomarker_file.endswith(biomarker_name.value + ".csv"):
                                df = pd.read_csv(biomarkers_path.joinpath(biomarker_file), sep=',')
                                df['datetime'] = pd.to_datetime(df['timestamp_iso'],
                                                                utc=True).map(lambda x: x.tz_convert('Asia/Jerusalem'))
                                df2 = df.dropna(subset=[biomarker_value_names[biomarker_name]])
                                df2 = df2[["datetime", biomarker_value_names[biomarker_name]]]
                                biomarker_dfs[biomarker_name] = pd.concat([biomarker_dfs[biomarker_name], df2])

                    # Process Raw EDA if needed
                    if Biomarker.EdaPhasic in biomarker_names or Biomarker.EdaTonic in biomarker_names:
                        raw_eda_dfs = []
                        raw_data_dir = user_path.joinpath("raw_data/v6")
                        if raw_data_dir.exists():
                            print(f"Processing raw EDA for {user_dir}...")
                            for avro_file in sorted(os.listdir(raw_data_dir)):
                                if avro_file.endswith(".avro"):
                                    try:
                                        eda_df, _, _ = avro_utils.generate_dataframes_from_avro(
                                            raw_data_dir.joinpath(avro_file), jerusalem_tz)
                                        # median_val = eda_df['value'].median()
                                        # if pd.notna(median_val) and median_val != 0:
                                        #     eda_df['value'] = eda_df['value'] / median_val
                                        eda_df['value'] = eda_df['value']
                                        if not eda_df.empty:
                                            raw_eda_dfs.append(eda_df)
                                    except Exception as e:
                                        print(f"Error processing {avro_file}: {e}")

                            if raw_eda_dfs:
                                full_eda_df = pd.concat(raw_eda_dfs).sort_values('datetime').drop_duplicates('datetime')
                                # NeuroKit2 processing
                                try:
                                    # Resample to a consistent rate if needed, but eda_process handles it usually if sampling_rate is provided.
                                    # However, our df has timestamps. nk.eda_process expects a signal.
                                    # We need to infer sampling rate or just pass the signal values.
                                    # EmbracePlus EDA is usually 4Hz (check?), but avro_utils extracts it.
                                    # Let's assume constant sampling rate from the first file or just use the extracted times.
                                    # nk.eda_process(eda_signal, sampling_rate=...)

                                    # Calculating sampling rate
                                    if len(full_eda_df) > 1:
                                        time_diffs = full_eda_df['datetime'].diff().dt.total_seconds().dropna()
                                        fs = 1 / time_diffs.median()

                                        signals, info = nk.eda_process(full_eda_df['value'], sampling_rate=fs)

                                        # signals contains EDA_Raw, EDA_Clean, EDA_Phasic, EDA_Tonic, etc.
                                        # Map back to datetime
                                        # signals length should match input

                                        if Biomarker.EdaPhasic in biomarker_names:
                                            phasic_df = pd.DataFrame({
                                                'datetime': full_eda_df['datetime'].reset_index(drop=True),
                                                biomarker_value_names[Biomarker.EdaPhasic]: signals['EDA_Phasic'].values
                                            })
                                            biomarker_dfs[Biomarker.EdaPhasic] = pd.concat(
                                                [biomarker_dfs[Biomarker.EdaPhasic], phasic_df])

                                        if Biomarker.EdaTonic in biomarker_names:
                                            tonic_df = pd.DataFrame({
                                                'datetime': full_eda_df['datetime'].reset_index(drop=True),
                                                biomarker_value_names[Biomarker.EdaTonic]: signals['EDA_Tonic'].values
                                            })
                                            biomarker_dfs[Biomarker.EdaTonic] = pd.concat(
                                                [biomarker_dfs[Biomarker.EdaTonic], tonic_df])
                                except Exception as e:
                                    print(f"Error in NeuroKit2 processing: {e}")

        # Save to cache
        print(f"Saving processed data to cache: {cache_file}")
        pd.to_pickle(biomarker_dfs, cache_file)
    return biomarker_dfs


if __name__ == '__main__':
    main()
