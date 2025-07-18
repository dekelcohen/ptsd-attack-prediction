import os
from pathlib import Path

import pandas as pd
import pytz
from pandas import DataFrame

from avro_utils import generate_dataframes_from_avro


def main():
    # user_id = "TRAIL002"
    # trial_starting_date = "2025-04-09 14:30:00"
    # user_id = "DEV008"
    # trial_starting_date = "2025-03-30 01:00:00"
    # user_id = "TRAIL001"
    # trial_starting_date = "2025-04-24 11:40:00"
    # user_id = "TRAIL004"
    # trial_starting_date = "2025-04-25 15:00:00"
    user_id = "TRAIL005"
    trial_starting_date = "2025-05-14 11:56:00"
    # user_id = "TRAIL003"
    # trial_starting_date = "2025-03-30 16:15:00"
    trial_starting_datetime = pd.to_datetime(trial_starting_date).tz_localize('Asia/Jerusalem')
    jerusalem_tz = pytz.timezone('Asia/Jerusalem')
    data_root_dir = Path("data/embrace_plus/")

    prepare_data_from_avro(data_root_dir, jerusalem_tz, trial_starting_date,
                           user_id)


def prepare_data_from_avro(data_root_dir, jerusalem_tz, trial_starting_datetime,
                           user_id):
    participant_data_dir = data_root_dir.joinpath("participant_data")
    participant_processed_data_dir = data_root_dir.joinpath("participant_processed_data")
    all_sps_df = DataFrame()
    all_eda_df = DataFrame()
    all_temp_df = DataFrame()
    invalid_minutes_df = set()
    for date_dir in os.listdir(participant_data_dir):
        date_path = participant_data_dir.joinpath(date_dir)
        for user_dir in os.listdir(date_path):
            if user_dir.startswith(user_id):
                print(f"processing user {user_dir} on date {date_dir}")
                user_processed_data_dir = participant_processed_data_dir.joinpath(user_id)
                if not user_processed_data_dir.exists():
                    user_processed_data_dir.mkdir()

                user_path = date_path.joinpath(user_dir)
                raw_data_path = user_path.joinpath("raw_data/")
                csv_dirs = os.listdir(os.path.join(raw_data_path, "v6"))
                for csv_dir in csv_dirs:
                    if csv_dir.endswith('avro'):
                        eda_df, temp_df, sps_df = generate_dataframes_from_avro(
                            os.path.join(raw_data_path, "v6", csv_dir),
                            jerusalem_tz)
                        if not eda_df.empty:
                            all_eda_df = pd.concat([all_eda_df, eda_df])
                        if not temp_df.empty:
                            all_temp_df = pd.concat([all_temp_df, temp_df])
                        if not sps_df.empty:
                            all_sps_df = pd.concat([all_sps_df, sps_df])
                biomarkers_path = user_path.joinpath("digital_biomarkers/aggregated_per_minute/")
                for biomarker_file in os.listdir(biomarkers_path):
                    if biomarker_file.endswith("pulse-rate" + ".csv"):
                        df = pd.read_csv(biomarkers_path.joinpath(biomarker_file), sep=',')
                        df['datetime'] = pd.to_datetime(df['timestamp_iso'],
                                                        utc=True).map(lambda x: x.tz_convert('Asia/Jerusalem'))
                        df = df[df['pulse_rate_bpm'].isnull()]
                        invalid_minutes_df.update(set(df['datetime'].dt.floor('min')))
                        break

    all_sps_df = all_sps_df[(all_sps_df['datetime'] >= trial_starting_datetime)]
    all_eda_df = all_eda_df[(all_eda_df['datetime'] >= trial_starting_datetime)]
    all_temp_df = all_temp_df[(all_temp_df['datetime'] >= trial_starting_datetime)]

    total_hr_df = calculate_hr(all_sps_df)
    clean_all_hr_df = remove_invalid_minutes(invalid_minutes_df, total_hr_df)
    all_eda_df = round_time_and_interpulate(all_eda_df)
    clean_all_eda_df = remove_invalid_minutes(invalid_minutes_df, all_eda_df)
    all_temp_df = round_time_and_interpulate(all_temp_df)
    clean_all_temp_df = remove_invalid_minutes(invalid_minutes_df, all_temp_df)
    clean_all_hr_df.to_parquet(user_processed_data_dir.joinpath("all_hr_df.parquet"), index=False)
    clean_all_eda_df.to_parquet(user_processed_data_dir.joinpath("all_eda_df.parquet"), index=False)
    clean_all_temp_df.to_parquet(user_processed_data_dir.joinpath("all_temp_df.parquet"), index=False)


def round_time_and_interpulate(df):
    df['rounded_timestamp'] = (df['datetime'].dt.floor('s') +
                               (df['datetime'].dt.microsecond // 250000) * pd.Timedelta('250ms'))
    min_timestamp = df['rounded_timestamp'].min()
    max_timestamp = df['rounded_timestamp'].max()
    all_quarters = pd.date_range(start=min_timestamp, end=max_timestamp, freq='250ms')
    # Merge the new DataFrame with the result DataFrame to get heart rate per minute for every quarter of a second
    full_result_df = pd.DataFrame({'rounded_timestamp': all_quarters})
    merged_result_df = full_result_df.merge(df, on='rounded_timestamp', how='left')
    # Interpolate the heart rate values to fill NaN values
    merged_result_df['value'] = merged_result_df['value'].interpolate()
    return merged_result_df


def remove_invalid_minutes(invalid_minutes_df, all_minutes_df):
    all_hr_df = all_minutes_df.copy()
    all_hr_df['minute_resolution'] = all_hr_df['rounded_timestamp'].dt.floor('min')
    all_hr_df = all_hr_df[~all_hr_df['minute_resolution'].isin(invalid_minutes_df)]
    # all_hr_df = all_hr_df.dropna()
    return all_hr_df


def calculate_hr(df: pd.DataFrame, timestamp_column: str = 'datetime') -> pd.DataFrame:

    # Calculate the time difference between consecutive systolic peaks in seconds
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()

    # Calculate the heart rate per minute for each systolic peak
    df['hr_per_minute'] = 60 / df['time_diff']

    # Forward fill the heart rate values to fill NaN values for the first row
    df['hr_per_minute'] = df['hr_per_minute'].ffill()

    # Round the timestamps to the nearest quarter of a second
    df['rounded_timestamp'] = (df['datetime'].dt.floor('s') +
                               (df['datetime'].dt.microsecond // 250000) * pd.Timedelta('250ms'))

    # Group by the rounded timestamp and calculate the average heart rate per minute for each quarter of a second
    result_df = df.groupby('rounded_timestamp')['hr_per_minute'].mean().reset_index()

    full_result_df = convert_to_4hz(result_df)
    return full_result_df


def convert_to_4hz(df):
    # Create a new DataFrame with a row for every quarter of a second in the input DataFrame
    min_timestamp = df['rounded_timestamp'].min()
    max_timestamp = df['rounded_timestamp'].max()
    all_quarters = pd.date_range(start=min_timestamp, end=max_timestamp, freq='250ms')
    # Merge the new DataFrame with the result DataFrame to get heart rate per minute for every quarter of a second
    full_result_df = pd.DataFrame({'rounded_timestamp': all_quarters})
    full_result_df = full_result_df.merge(df, on='rounded_timestamp', how='left')
    # Interpolate the heart rate values to fill NaN values
    full_result_df['hr_per_minute'] = full_result_df['hr_per_minute'].interpolate()
    return full_result_df
