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


def df_timestamp_to_israel_time(df, timestamp_col):
    """
    Convert a timestamp column in a DataFrame to Israel local time (Asia/Jerusalem).

    The function accepts either tz-naive or tz-aware timestamps:
    - If the column is tz-naive, it is interpreted as UTC, localized to UTC,
      and then converted to Asia/Jerusalem.
    - If the column is tz-aware, it is directly converted to Asia/Jerusalem.

    A new column named ``timestamp_israel`` is added to ``df`` with the converted
    timezone-aware timestamps.

    Args:
        df (pandas.DataFrame): Input DataFrame containing a timestamp column.
        timestamp_col (str): Name of the timestamp column to convert.

    Returns:
        pandas.DataFrame: The same DataFrame with an added ``timestamp_israel`` column
        (dtype: datetime64[ns, Asia/Jerusalem]). The original ``timestamp_col`` is parsed
        to datetime if needed.

    Notes:
        - This function assumes tz-naive values are in UTC.
        - The output column is timezone-aware (not naive).
    """
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    if df[timestamp_col].dt.tz is None:
        # tz-naive → localize first
        df['timestamp_israel'] = df[timestamp_col].dt.tz_localize('UTC').dt.tz_convert('Asia/Jerusalem')
    else:
        # already tz-aware → just convert
        df['timestamp_israel'] = df[timestamp_col].dt.tz_convert('Asia/Jerusalem')
    return df


def create_biomarkers_data_for_patient(path, patients_dict, data, patient):
    """
    Aggregate per-minute digital biomarker CSVs for a single patient across all days.

    The function walks the directory structure under ``path`` expecting a layout like:
        <path>/<day>/<patient_folder>/digital_biomarkers/aggregated_per_minute/*.csv

    It filters for a predefined set of biomarker files and merges them on
    ``timestamp_israel`` after converting each file's ``timestamp_iso`` to Israel time.
    Columns used only for alignment/metadata are excluded from duplicate merges.

    Args:
        path (str): Root directory containing day folders.
        patients_dict (dict): Mapping from patient short ID to folder name, e.g.
            ``{'P01': 'participant_0001'}``.
        data (pandas.DataFrame): Initial DataFrame to append results to (can be empty).
        patient (str): Patient short ID (key in ``patients_dict``) to process.

    Returns:
        pandas.DataFrame: Concatenated DataFrame of per-minute biomarkers for the
        specified patient across all available days. Includes a timezone-aware
        ``timestamp_israel`` column.

    Expected Input Columns (per CSV):
        - ``timestamp_iso`` (ISO-8601 string) used for time conversion.
        - Optional metadata: ``participant_full_id``, ``timestamp_unix``,
          ``missing_value_reason`` (excluded from merge when duplicative).
    """
    biomarker_file_names = ["eda.csv", "temperature.csv", "accelerometers-std.csv",
                            "pulse-rate.csv", "activity-counts.csv"]

    for day in os.listdir(path):
        join_path = os.path.join(path, day)
        for temp_patient in os.listdir(join_path):
            if temp_patient == patients_dict[patient]:
                day_biomarkers = pd.DataFrame()
                digital_biomarker_path = os.path.join(join_path, patients_dict[patient],
                                                      r'digital_biomarkers\aggregated_per_minute')
                for file in os.listdir(digital_biomarker_path):
                    if file.split('_')[-1] not in biomarker_file_names:
                        continue
                    df = pd.read_csv(os.path.join(digital_biomarker_path, file))
                    df = df_timestamp_to_israel_time(df, timestamp_col='timestamp_iso')
                    if day_biomarkers.empty:
                        day_biomarkers = df
                    else:
                        exclude = {'participant_full_id', 'timestamp_iso', 'timestamp_unix',
                                   'missing_value_reason'}

                        cols_from_df = [c for c in df.columns if c not in exclude]

                        day_biomarkers = day_biomarkers.merge(
                            df[cols_from_df],
                            on="timestamp_israel",
                            how="left"
                        )
                data = pd.concat([data, day_biomarkers])
    data = data[~data['missing_value_reason'].isin(['device_not_recording', 'device_not_worn_correctly'])]
    return data


def filter_biomarkers_data_around_tags_for_patient(tags, data, time='15min'):
    """
    Label biomarker rows by the nearest positive-severity tag within a time tolerance.

    For each row in ``data``, the function finds the nearest event in ``tags`` within
    ``tolerance = pd.Timedelta(time)`` (default 15 minutes), using the
    ``timestamp_israel`` index. Only tags with ``severity > 0`` are considered.
    The result is split into two DataFrames:
      - ``df_selected``: rows that matched a nearby tag (with added ``eventType`` and ``severity``)
      - ``df_remaining``: rows without a nearby tag

    Args:
        tags (pandas.DataFrame): Tag data containing at least
            ``['timestamp_israel', 'eventType', 'severity']``.
        data (pandas.DataFrame): Biomarker data containing ``timestamp_israel``.
        time (str | pandas.Timedelta, optional): Time tolerance for nearest-match
            (e.g., '15min', '5m', '1H'). Defaults to '15min'.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:
            (df_selected, df_remaining) as described above.
    """
    positive_tags = tags[tags['severity'] > 0]
    # Set index to timestamp for both eventType and severity
    tags_by_time = positive_tags.set_index('timestamp_israel')[['eventType', 'severity']]

    # Remove duplicate timestamps, keeping the last occurrence
    tags_by_time = tags_by_time[~tags_by_time.index.duplicated(keep='last')]

    # Find nearest events for both columns
    nearest_events = tags_by_time.reindex(
        data['timestamp_israel'],
        method='nearest',
        tolerance=pd.Timedelta(time)
    )

    # Copy data and add both eventType and severity columns
    data_with_event = data.copy()
    data_with_event['eventType'] = nearest_events['eventType'].values
    data_with_event['severity'] = nearest_events['severity'].values

    # rows WITH matching tag (where eventType is not null)
    df_selected = data_with_event[data_with_event['eventType'].notna()]

    # rows WITHOUT matching tag (where eventType is null)
    df_remaining = data_with_event[data_with_event['eventType'].isna()]

    return df_selected, df_remaining


def prepare_biomarkers_data(patients_dict, tags_path, data_path, time='15min'):
    """
    Build labeled biomarker datasets by (a) aggregating per-patient biomarker streams
    and (b) aligning them to nearby positive-severity tags.

    For each ``<patient>_*.csv`` file in ``tags_path``:
      1) Load tags, convert their timestamps to Israel time.
      2) Aggregate that patient's biomarker per-minute CSVs from ``data_path``.
      3) Match biomarker rows to the nearest tag within ``time`` tolerance.
      4) Concatenate matched rows across all patients/days, and also keep unmatched rows.

    Args:
        patients_dict (dict): Mapping from patient short ID to folder name in the data root.
        tags_path (str): Directory containing per-patient tag CSV files. Filenames
            should start with the patient key, e.g., ``P01_events.csv``.
        data_path (str): Root directory for biomarker data (see
            ``create_biomarkers_data_for_patient`` for expected structure).
        time (str | pandas.Timedelta, optional): Tolerance for tag matching. Defaults to '15min'.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]:
            - ``filtered_around_tags_data``: biomarker rows matched to nearby positive tags,
              with ``eventType`` and ``severity`` attached.
            - ``remaining_data``: biomarker rows with no nearby positive tag.
    """
    filtered_around_tags_data = pd.DataFrame()
    remaining_data = pd.DataFrame()
    total_number_of_tags = 0
    total_number_of_tags_per_patient = {}
    for file in os.listdir(tags_path):
        if file.endswith(".csv"):
            patient = file.split("_")[0]
            if patient in patients_dict.keys():
                tags = pd.read_csv(os.path.join(tags_path, file))
                # tags = tags[tags['eventType'] != 'other']
                tags = df_timestamp_to_israel_time(tags, timestamp_col='timestamp')
                total_number_of_tags += len(tags)
                total_number_of_tags_per_patient[patient] = len(tags)
                data = pd.DataFrame()
                data = create_biomarkers_data_for_patient(data_path, patients_dict, data, patient)

                filtered_data, other_data = filter_biomarkers_data_around_tags_for_patient(tags, data, time)
                filtered_around_tags_data = pd.concat([filtered_around_tags_data, filtered_data])
                remaining_data = pd.concat([remaining_data, other_data])
    print('Total number of tags: ', total_number_of_tags)
    print('Total number of tags per patient: \n', total_number_of_tags_per_patient)
    return filtered_around_tags_data, remaining_data
