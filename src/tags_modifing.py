
import os
import sys
import pandas as pd
import pytz
from datetime import timedelta
from pathlib import Path
from data_visualization import prepare_data, detect_anomalies
from biomarkers import Biomarker
import openpyxl
from utils import load_participant_dates


def get_start_date(user_id, participant_dates):
    # Try the dictionary
    if user_id in participant_dates:
        return pd.to_datetime(participant_dates[user_id]['start_date'], dayfirst=True)
    # Default fallback
    print(f"Warning: No start date found for {user_id}, using default.")
    return pd.to_datetime("2024-01-01")


def modify_tag_timestamps(user_id, raw_tags_path=None, output_path=None, 
                          min_allowed_time_hours=6, max_allowed_time_hours=0.5, cluster_gap_minutes=30,
                          participant_dates=None):
    
    jerusalem_tz = pytz.timezone('Asia/Jerusalem')
    data_root_dir = Path("data/embrace_plus/")
    
    # 1. Determine paths
    if raw_tags_path is None:
        raw_tags_path = data_root_dir / f"participants_extra_data/valid_tags/{user_id}_valid_tags_raw.csv"
    
    if output_path is None:
        output_path = data_root_dir / f"participants_extra_data/valid_tags/{user_id}_valid_tags_modified.csv"
        
    if not os.path.exists(raw_tags_path):
        # Fallback to normal valid tags if raw doesn't exist
        print(f"Raw tags file not found at {raw_tags_path}, trying standard valid tags.")
        raw_tags_path = data_root_dir / f"participants_extra_data/valid_tags/{user_id}_valid_tags.csv"
        if not os.path.exists(raw_tags_path):
             print(f"No tags file found for {user_id}. Exiting.")
             return

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Processing tags for {user_id}...")
    print(f"Reading tags from: {raw_tags_path}")
    
    # 2. Load Biomarker Data & Anomalies
    trial_start_date = get_start_date(user_id, participant_dates)
    # Format for prepare_data is expected to be string or datetime
    # prepare_data expects a string usually? visualize_main passes string.
    # But it calls pd.to_datetime inside.
    
    # biomarker_names = [Biomarker.Pr, Biomarker.Eda, Biomarker.EdaPhasic, Biomarker.EdaTonic,
    #                    Biomarker.AccStd, Biomarker.Prv, Biomarker.Met, Biomarker.Temp]

    biomarker_names = [Biomarker.Pr, Biomarker.AccStd, Biomarker.Temp,
                       Biomarker.EdaPhasic, Biomarker.EdaTonic]

    print("Loading biomarker data...")
    # Passing trial_start_date as string to stay safe with prepare_data signature expecting potentially string
    biomarker_dfs = prepare_data(biomarker_names, data_root_dir, jerusalem_tz, None, str(trial_start_date), user_id)
    
    print("Detecting anomalies...")
    anomaly_df = detect_anomalies(biomarker_dfs)
    
    if anomaly_df.empty:
        print("No anomalies detected. Saving tags as is.")
        df_tags = pd.read_csv(raw_tags_path)
        df_tags['new_timestamp'] = df_tags['timestamp']
        df_tags.to_csv(output_path, index=False)
        return

    # Ensure anomaly timestamps are localized/converted to match tags
    # detect_anomalies returns 'datetime' column. 
    # prepare_data converts biomarkers to Asia/Jerusalem.
    # anomaly_df should share this timezone.
    
    # 3. Process Tags
    df_tags = pd.read_csv(raw_tags_path)
    
    # Parse tag timestamps
    # Handling " IDT" per visualization code
    df_tags["timestamp_clean"] = df_tags["timestamp"].astype(str).str.replace(" IDT", "").str.replace(" IST", "")
    # Assume tags are in local time (Jerusalem)
    try:
        tag_datetimes = pd.to_datetime(df_tags['timestamp_clean']).dt.tz_localize(jerusalem_tz, ambiguous='NaT', nonexistent='NaT')
    except Exception:
         # If already tz aware or other format issues
         tag_datetimes = pd.to_datetime(df_tags['timestamp_clean'], utc=True).dt.tz_convert(jerusalem_tz)
         
    new_timestamps = []
    
    anom_times = anomaly_df['datetime'].sort_values().reset_index(drop=True)
    
    for i, tag_time in enumerate(tag_datetimes):
        if pd.isna(tag_time) or df_tags.iloc[i]["severity"] < 1:
            timestamp_clean = df_tags.iloc[i]["timestamp"].replace(" IDT", "").replace(" IST", "")

            new_timestamps.append(timestamp_clean)  # Keep original if parse fail
            continue
            
        # Search window
        min_allowed_time = tag_time - timedelta(hours=min_allowed_time_hours)
        max_allowed_time = tag_time + timedelta(hours=max_allowed_time_hours)
        
        # Anomalies in window [min_allowed_time, max_allowed_time]
        candidates = anom_times[(anom_times >= min_allowed_time) & (anom_times <= max_allowed_time)]
        
        if candidates.empty:
            timestamp_clean = df_tags.iloc[i]["timestamp"].replace(" IDT", "").replace(" IST", "")

            new_timestamps.append(timestamp_clean)  # Keep original if parse fail

            continue
            
        # Closest to tag_time
        # Use abs diff
        closest_anomaly = candidates.iloc[(candidates - tag_time).abs().argmin()]
        
        # Walk back logic from closest_anomaly
        # Iterate backwards from closest_anomaly
        # We need the index of closest_anomaly in the full candidates logic?
        # Actually the 'walk back' logic implies finding earlier connected anomalies.
        # If closest_anomaly is AFTER tag_time, do we walk back?
        # Yes, standard logic: find an anchor (closest), then walk back to find the start of the cluster.
        
        current_marker = closest_anomaly
        
        # We need to search in the FULL anom_times or just candidates?
        # Typically walk back goes as far as the cluster allows.
        # Let's search in candidates + earlier ones?
        # To be safe and consistent with previous logic, we walk back using ALL available anomalies (sorted)
        # Find index of current_marker in anom_times
        
        # Since anom_times is sorted, we can search sorted
        # Or just use the timestamp value and find preceeding ones.
        
        # Find position in full sorted list
        # We can find the index efficiently
        # Since timestamps are unique (hopefully) or we just take the first match
        
        # Optimization: Filter anom_times to look only backward from current_marker
        # We only care about anomalies BEFORE current_marker
        
        possible_precursors = anom_times[anom_times <= current_marker]
        # Reverse iterate
        precursors_list = possible_precursors.tolist()
        
        modified_time = current_marker
        
        # Start from end-1 (before current_marker)
        for j in range(len(precursors_list) - 2, -1, -1):
            prev_anom = precursors_list[j]
            diff = modified_time - prev_anom
            if diff <= timedelta(minutes=cluster_gap_minutes):
                modified_time = prev_anom
            else:
                # Chain broken
                break
                
        # Format back to string or whatever format is desired
        # The output requested is a column "new_timestamp".
        # Let's clean formatting to be consistent with input or standard ISO
        # Input format seems custom ("... IDT").
        # Let's keep it defined as the datetime object string for now
        new_timestamps.append(modified_time.strftime("%Y-%m-%d %H:%M:%S"))

    df_tags['new_timestamp'] = new_timestamps
    
    # Deduplicate based on priority: app > remote > watch
    if 'origin' in df_tags.columns:
        # Map origin to priority (lower is better)
        df_tags['temp_priority'] = df_tags['origin'].astype(str).str.lower().map({
            'app': 1,
            'remote': 2,
            'watch': 3
        }).fillna(4)

        # Sort: first by time, then by priority (ascending)
        df_tags = df_tags.sort_values(by=['new_timestamp', 'temp_priority'], ascending=[True, True])

        # Drop duplicates keeping the first (highest priority)
        df_tags = df_tags.drop_duplicates(subset=['new_timestamp'], keep='first')

        # Drop the temp column
        df_tags = df_tags.drop(columns=['temp_priority'])
    
    # Handle Output Path
    out_p = Path(output_path)
    if out_p.is_dir():
        # If directory, append filename
        out_p = out_p / f"{user_id}_valid_tags_modified.csv"
    
    # Save
    df_tags.drop(columns=['timestamp_clean'], inplace=True, errors='ignore')
    print(f"Saving modified tags to {out_p}")
    df_tags.to_csv(out_p, index=False)


if __name__ == "__main__":
    # Load participant dates from Excel
    excel_path = r"data/embrace_plus/participants_extra_data/participant_data_periods.xlsx"
    participant_dates = load_participant_dates(excel_path)
    
    # Paths
    raw_tags_dir = r"data/embrace_plus/participants_extra_data/valid_tags/raw_tags"
    output_tags_dir = r"data/embrace_plus/participants_extra_data/valid_tags/auto_modified_tags"
    
    # Ensure output directory exists
    os.makedirs(output_tags_dir, exist_ok=True)

    # List all raw tag files
    raw_files = [f for f in os.listdir(raw_tags_dir) if f.endswith("_valid_tags_raw.csv")]
    
    for raw_file in raw_files:
        # Extract user_id, e.g. "TRAIL10_valid_tags_raw.csv" -> "TRAIL10"
        user_id = raw_file.replace("_valid_tags_raw.csv", "")

        raw_file_path = os.path.join(raw_tags_dir, raw_file)

        output_file_name = f"{user_id}_valid_tags_modified.csv"
        output_file_path = os.path.join(output_tags_dir, output_file_name)
        
        if os.path.exists(output_file_path):
            print(f"Skipping {user_id}: Output file {output_file_name} already exists.")
            continue

        print(f"\n--- Batch processing: {user_id} ---")
        try:
            modify_tag_timestamps(user_id, raw_tags_path=raw_file_path, output_path=output_tags_dir,
                                  participant_dates=participant_dates)
        except Exception as e:
            print(f"Failed to process {user_id}: {e}")

