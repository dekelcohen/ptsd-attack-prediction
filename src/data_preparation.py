import os
from pathlib import Path
from datetime import datetime

import numpy as np
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


def create_biomarkers_data_for_patient(path, patients_dict, data, patient, trail_dates_dict):
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
        day_date = datetime.strptime(day, "%Y-%m-%d").date()
        if day_date < trail_dates_dict[patient]['start_date'].date() or day_date > trail_dates_dict[patient]['end_date'].date():
            continue
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
    tags_by_time = tags_by_time.sort_index(kind='mergesort')  # ✅ critical for 'nearest'

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


def _robust_z(x):
    x = pd.to_numeric(x, errors="coerce")
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return (x - med) / (mad + 1e-6)


def _cos_night_prior(ts, night_center_hour=3, night_width_hours=10):
    """
    Smooth circadian prior in [0,1], peaking at `night_center_hour`, wide ~ `night_width_hours`.
    Uses a raised cosine: prior = clip(0.5*(1 + cos(2π*(Δhour)/width)), 0, 1)
    where Δhour is wrapped difference between local hour and night_center_hour.
    """
    hours = ts.dt.hour + ts.dt.minute / 60.0
    # Wrap difference into [-12,12]
    delta = ((hours - night_center_hour + 12) % 24) - 12
    prior = 0.5 * (1 + np.cos(np.pi * np.clip(delta / (night_width_hours / 2), -1, 1)))
    return prior


def _label_sleep_awake(
        df,
        time_col="timestamp_israel",
        hr_col="pulse_rate_bpm",
        activity_cols=None,  # e.g., ["activity_counts", "accelerometers_std_g"]
        w_hr=0.7,  # weight for HR evidence
        w_time=0.3,  # weight for time-of-day prior
        night_center_hour=3,
        night_width_hours=10,
        min_sleep_bout_minutes=20,  # post-processing to reduce flicker
        hr_clip_z=(-3, 3),  # clip HR z to stabilize
):
    """
    Adds a 'state' column with 'sleep'/'awake' per subject.
    Heuristic: low HR (subject-robust z) + night prior => sleep.
    Optionally nudged by low movement if activity_cols provided.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # We'll build per-subject scores
    states = []
    # for sid, g in df.sort_values(time_col):
    #     gg = g.copy()
    df = df.sort_values(time_col)

    # HR evidence (low HR -> sleep). Convert robust z to a [0,1] "sleepiness" via sigmoid.
    z_hr = _robust_z(df[hr_col])
    z_hr = np.clip(z_hr, hr_clip_z[0], hr_clip_z[1])
    # Map: very low z -> near 1, high z -> near 0
    hr_sleepiness = 1 / (1 + np.exp(1.5 * z_hr))  # 1.5 slope works well; tweak if needed

    # Optional movement evidence (low activity -> sleep)
    if activity_cols:
        activity_sleepiness_list = []
        for c in activity_cols:
            z_act = _robust_z(gg[c])
            # low activity => sleep: invert sign
            activity_sleepiness_list.append(1 / (1 + np.exp(1.5 * z_act)))
        act_sleepiness = np.nanmean(np.vstack(activity_sleepiness_list), axis=0)
        # If present, blend HR and activity first (lean on HR)
        evidence = 0.8 * hr_sleepiness + 0.2 * act_sleepiness
    else:
        evidence = hr_sleepiness

    # Time-of-day prior (smooth, not hard). In [0,1].
    time_prior = _cos_night_prior(df[time_col], night_center_hour, night_width_hours)

    # Final score: weighted combination
    score = w_hr * evidence + w_time * time_prior

    # Auto-threshold per subject via Otsu-like split on score histogram;
    # fallback to 0.5 if degenerate.
    s = score[np.isfinite(score)]
    if len(s) > 12:
        hist, bins = np.histogram(s, bins=32, range=(0, 1))
        p = hist.astype(float) / (hist.sum() + 1e-9)
        omega = np.cumsum(p)
        mu = np.cumsum(p * ((bins[:-1] + bins[1:]) / 2))
        mu_t = mu[-1]
        sigma_b2 = (mu_t * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
        k = np.nanargmax(sigma_b2)
        thr = (bins[k] + bins[k + 1]) / 2
    else:
        thr = 0.5

    raw_state = np.where(score >= thr, 1, 0)  # 1=sleep, 0=awake

    # Post-process: enforce minimum sleep bout length (in minutes, given per-minute sampling)
    # Convert short isolated sleep islands to awake
    arr = raw_state.copy()
    if min_sleep_bout_minutes and len(arr) > 0:
        # Find segments of consecutive 1's
        diff = np.diff(np.r_[0, arr, 0])
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for st, en in zip(starts, ends):
            if (en - st) < min_sleep_bout_minutes:
                arr[st:en] = 0

    df["state"] = np.where(arr == 1, "sleep", "awake")
    states.append(df)

    out = pd.concat(states, axis=0).sort_index()
    return out


def normalize_by_subject_and_state(
        df,
        biomarker_cols=None,
        time_col="timestamp_israel",
        hr_col="pulse_rate_bpm",
        activity_cols=None,
        **sleep_kwargs,
):
    """
    1) infers 'state' per row (sleep/awake)
    2) robust-normalizes each biomarker within (subject, state)
    Returns a new DataFrame with normalized biomarker columns (overwriting the originals)
    and a 'state' column.
    """
    if biomarker_cols is None:
        biomarker_cols = [
            "accelerometers_std_g",
            "activity_counts",
            "eda_scl_usiemens",
            "pulse_rate_bpm",
            "temperature_celsius",
        ]
    df = df.copy()
    df = _label_sleep_awake(
        df,
        time_col=time_col,
        hr_col=hr_col,
        activity_cols=activity_cols,
        **sleep_kwargs,
    )

    # Normalize within (subject, state)
    def _normalize_group(g):
        g = g.copy()
        for c in biomarker_cols:
            x = pd.to_numeric(g[c], errors="coerce")
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            g[c] = (x - med) / (mad + 1e-6)
        return g

    df = (
        df.sort_values(time_col)
        .groupby("state", group_keys=False)
        .apply(_normalize_group)
        .reset_index(drop=True)
    )
    return df


def prepare_biomarkers_data(patients_dict, tags_path, data_path, time='15min', trail_dates_dict=None, normalize=False):
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
                data = create_biomarkers_data_for_patient(data_path, patients_dict, data, patient, trail_dates_dict)
                if normalize:
                    data = normalize_by_subject_and_state(data)
                filtered_data, other_data = filter_biomarkers_data_around_tags_for_patient(tags, data, time)
                filtered_around_tags_data = pd.concat([filtered_around_tags_data, filtered_data])
                remaining_data = pd.concat([remaining_data, other_data])
    print('Total number of tags: ', total_number_of_tags)
    print('Total number of tags per patient: \n', total_number_of_tags_per_patient)
    return filtered_around_tags_data, remaining_data


def _flatten_cols(df):
    df.columns = [
        "_".join([c for c in map(str, col) if c != "" and c is not None])
        if isinstance(col, tuple) else str(col)
        for col in df.columns
    ]
    return df


import numpy as np
import pandas as pd

EPS = 1e-6  # for MAD denominator safety
import numpy as np
import pandas as pd

EPS = 1e-6  # for MAD denominator safety


def _tail_indices_non_na(s: pd.Series, k: int):
    m = s.notna()
    if not m.any():
        return []
    idx = np.flatnonzero(m.values)
    return list(idx[-min(k, len(idx)):])


def _slope_tail_per_min(s: pd.Series, k: int) -> float:
    """
    Linear-fit slope over the last up-to-k non-NaN points (units per minute).
    Uses true timestamps if the index is datetime-like (handles tz-aware).
    """
    tail_pos = _tail_indices_non_na(s, k)
    if len(tail_pos) < 2:
        return np.nan

    y = s.iloc[tail_pos].astype(float).values
    idx = s.index[tail_pos]

    # Build minutes axis
    if isinstance(idx, pd.DatetimeIndex):
        # Convert tz-aware -> UTC -> tz-naive to get int64 ns
        if idx.tz is not None:
            idx_naive = idx.tz_convert("UTC").tz_localize(None)
        else:
            idx_naive = idx
        t_min = (idx_naive.view("int64") - idx_naive.view("int64")[0]) / 60_000_000_000.0
    else:
        # assume 1-min cadence
        t_min = np.arange(len(y), dtype=float)

    if np.allclose(t_min, t_min[0]):
        return np.nan

    slope, _ = np.polyfit(t_min, y, 1)
    return float(slope)


def _mad(x: np.ndarray) -> float:
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def _window_features(chunk: pd.DataFrame, value_cols):
    out = {}
    for c in value_cols:
        s = pd.to_numeric(chunk[c], errors="coerce")

        n_valid = int(s.count())
        n_total = int(s.size)
        n_na = n_total - n_valid

        out[(c, "count")] = n_valid
        out[(c, "n_na")] = n_na
        out[(c, "na_ratio")] = (n_na / n_total) if n_total > 0 else np.nan

        if n_valid == 0:
            for k in ["mean", "std", "min", "q25", "median", "q75", "max",
                      "skew", "kurt", "first", "last", "range", "delta", "slope_per_s",
                      "ema30_end", "dev1_end", "dev5_end", "dev15_end",
                      "slope5_per_min", "slope15_per_min", "range5", "z15_end"]:
                out[(c, k)] = np.nan
            continue

        # basic stats
        out[(c, "mean")] = s.mean()
        out[(c, "std")] = s.std(ddof=1) if n_valid >= 2 else np.nan
        out[(c, "min")] = s.min()
        out[(c, "q25")] = s.quantile(0.25)
        out[(c, "median")] = s.median()
        out[(c, "q75")] = s.quantile(0.75)
        out[(c, "max")] = s.max()

        # shape stats
        out[(c, "skew")] = s.skew() if n_valid >= 3 else np.nan
        out[(c, "kurt")] = s.kurt() if n_valid >= 4 else np.nan

        # edges
        s_non = s.dropna()
        first = s_non.iloc[0] if not s_non.empty else np.nan
        last = s_non.iloc[-1] if not s_non.empty else np.nan
        out[(c, "first")] = first
        out[(c, "last")] = last
        out[(c, "range")] = out[(c, "max")] - out[(c, "min")]
        out[(c, "delta")] = (last - first) if pd.notna(last) and pd.notna(first) else np.nan

        # ----- FIXED: slope over whole chunk (per second), tz-safe -----
        m = s.notna().values
        if m.sum() >= 2:
            idx = chunk.index
            if isinstance(idx, pd.DatetimeIndex):
                # Convert tz-aware -> UTC -> tz-naive
                if idx.tz is not None:
                    idx_naive = idx.tz_convert("UTC").tz_localize(None)
                else:
                    idx_naive = idx
                # float seconds since epoch for the valid points
                t_sec = (idx_naive.view("int64")[m] / 1e9).astype(float)
            else:
                # assume 1-min cadence -> seconds grid
                t_sec = (np.arange(len(idx))[m] * 60.0).astype(float)

            y = s[m].astype(float).values
            if np.unique(t_sec).size >= 2:
                slope, _ = np.polyfit(t_sec, y, 1)
                out[(c, "slope_per_s")] = float(slope)
            else:
                out[(c, "slope_per_s")] = np.nan
        else:
            out[(c, "slope_per_s")] = np.nan

        # =========== NEW end-of-window sequence features ===========
        tail5_pos = _tail_indices_non_na(s, 5)
        tail15_pos = _tail_indices_non_na(s, 15)

        ema30_series = s.ewm(span=30, adjust=False, min_periods=1).mean()
        ema30_end = float(ema30_series.iloc[-1]) if not ema30_series.empty else np.nan
        out[(c, "ema30_end")] = ema30_end

        if pd.notna(last):
            mean5 = float(np.nanmean(s.iloc[tail5_pos].values)) if len(tail5_pos) > 0 else np.nan
            mean15 = float(np.nanmean(s.iloc[tail15_pos].values)) if len(tail15_pos) > 0 else np.nan

            out[(c, "dev1_end")] = float(last) - ema30_end if pd.notna(ema30_end) else np.nan
            out[(c, "dev5_end")] = (mean5 - ema30_end) if (pd.notna(mean5) and pd.notna(ema30_end)) else np.nan
            out[(c, "dev15_end")] = (mean15 - ema30_end) if (pd.notna(mean15) and pd.notna(ema30_end)) else np.nan
        else:
            out[(c, "dev1_end")] = out[(c, "dev5_end")] = out[(c, "dev15_end")] = np.nan

        out[(c, "slope5_per_min")] = _slope_tail_per_min(s, 5)
        out[(c, "slope15_per_min")] = _slope_tail_per_min(s, 15)

        if len(tail5_pos) > 0:
            tail5_vals = s.iloc[tail5_pos].values.astype(float)
            out[(c, "range5")] = float(np.nanmax(tail5_vals) - np.nanmin(tail5_vals))
        else:
            out[(c, "range5")] = np.nan

        if len(tail15_pos) > 0 and pd.notna(last):
            tail15_vals = s.iloc[tail15_pos].values.astype(float)
            med15 = float(np.nanmedian(tail15_vals))
            mad15 = _mad(tail15_vals)
            out[(c, "z15_end")] = (float(last) - med15) / (mad15 + EPS) if (
                    pd.notna(med15) and pd.notna(mad15)) else np.nan
        else:
            out[(c, "z15_end")] = np.nan
        # ===========================================================

    return out


def make_time_windows(df, ts_col, window_minutes, step_minutes=None, value_cols=None, group_col=None, label_func=None):
    assert ts_col in df.columns, f"{ts_col=} not in df"
    tmp = df.copy()
    tmp[ts_col] = pd.to_datetime(tmp[ts_col])
    tmp = tmp.sort_values([group_col, ts_col] if group_col else ts_col)
    tmp = tmp.set_index(ts_col)

    # choose value columns
    if value_cols is None:
        value_cols = tmp.select_dtypes(include=[np.number, "float", "int", "Int64"]).columns.tolist()
        if group_col and group_col in value_cols:
            value_cols.remove(group_col)
        value_cols.remove('timestamp_unix')
        value_cols.remove('severity')

    window = pd.Timedelta(minutes=window_minutes)
    step = pd.Timedelta(minutes=step_minutes) if step_minutes is not None else window

    # generate window starts
    start = tmp.index.min().ceil(step)  # start aligned forward to step grid
    end = tmp.index.max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame()

    starts = pd.date_range(start=start, end=end, freq=step)
    rows = []
    labels = []

    for s0 in starts:
        s1 = s0 + window
        chunk = tmp.loc[(tmp.index >= s0) & (tmp.index < s1)]
        if chunk.empty:
            continue
        feats = _window_features(chunk, value_cols)
        # metadata
        feats[("timestamp_israel")] = s0
        # feats[("meta", "start")] = s0
        # feats[("meta", "end")] = s1
        # feats[("meta", "n_rows")] = len(chunk)
        rows.append(feats)
        if label_func is not None:
            labels.append(label_func(chunk))

    if not rows:
        return pd.DataFrame()

    X = pd.DataFrame(rows)
    X = _flatten_cols(X)
    if label_func is not None:
        y = pd.Series(labels, name="label").reset_index(drop=True)
        X = pd.concat([X, y], axis=1)

    # split X / y if label exists
    if "label" in X.columns:
        y = X.pop("label")
        return X, y
    return X, None


def create_chunked_data(data, patients_dict, window_minutes=5, step_minutes=3):
    res = pd.DataFrame()
    data = data.drop(columns=['missing_value_reason'])
    for patient in patients_dict.keys():
        patient_data = data[data['participant_full_id'].str.contains(patient, na=False)]
        patient = patient_data['participant_full_id'].unique()
        patient_data, _ = make_time_windows(df=patient_data, ts_col='timestamp_israel',
                                            window_minutes=window_minutes, step_minutes=step_minutes)
        patient_data['participant_full_id'] = patient[0]
        res = pd.concat([res, patient_data])
    return res


if __name__ == '__main__':
    patients_dict = {
        'TRAIL001': 'TRAIL001-3YK3L151K2',
        'TRAIL002': 'TRAIL002-3YK3J1514F',
        'TRAIL003': 'TRAIL003-3YK3K153QJ',
        'TRAIL004': 'TRAIL004-3YK3J151CV',
        'TRAIL005': 'TRAIL005-3YK3L151DR'}
    eval_patients_dict = {
        'TRAIL008': 'TRAIL008-3YK3J1514F',
        'TRAIL009': 'TRAIL009-3YKC51P1YL'
    }
    time = '15min'

    window_minutes_list = [5, 7, 10, 15]
    step_minutes_list = [1, 3]

    window_minutes_list = [15]
    step_minutes_list = [1]
    #
    # for window_minutes in window_minutes_list:
    #     for step_minutes in step_minutes_list:
    #         tags_path = r'../data\embrace_plus\participants_extra_data\valid_tags'
    #         data_path = r'C:\Users\GONY\Desktop\Booggii\data'
    #         chunked_data_path = fr'C:\Users\GONY\Desktop\Booggii\processed_data\{window_minutes}min_{step_minutes}step'
    #
    #         os.makedirs(chunked_data_path, exist_ok=True)
    #         print('creating data')
    #         positive_data, negative_data = prepare_biomarkers_data(patients_dict, tags_path, data_path, time)
    #         negative_data = create_chunked_data(negative_data, patients_dict, window_minutes=window_minutes,
    #                                             step_minutes=step_minutes)
    #         positive_data = create_chunked_data(positive_data, patients_dict, window_minutes=window_minutes,
    #                                             step_minutes=step_minutes)
    #         positive_data.to_pickle(
    #             chunked_data_path + rf'\train_eval_positive_data_{window_minutes}min_{step_minutes}step.pkl')
    #         negative_data.to_pickle(
    #             chunked_data_path + rf'\train_eval_negative_data_{window_minutes}min_{step_minutes}step.pkl')
    #
    #         eval_positive_data, eval_negative_data = prepare_biomarkers_data(eval_patients_dict, tags_path, data_path,
    #                                                                          time)
    #         eval_negative_data = create_chunked_data(eval_negative_data, eval_patients_dict,
    #                                                  window_minutes=window_minutes,
    #                                                  step_minutes=step_minutes)
    #         eval_positive_data = create_chunked_data(eval_positive_data, eval_patients_dict,
    #                                                  window_minutes=window_minutes,
    #                                                  step_minutes=step_minutes)
    #         eval_positive_data.to_pickle(
    #             chunked_data_path + rf'\test_positive_data_{window_minutes}min_{step_minutes}step.pkl')
    #         eval_negative_data.to_pickle(
    #             chunked_data_path + rf'\test_negative_data_{window_minutes}min_{step_minutes}step.pkl')

    for window_minutes in window_minutes_list:
        for step_minutes in step_minutes_list:
            tags_path = r'../data\embrace_plus\participants_extra_data\valid_tags'
            data_path = r'C:\Users\GONY\Desktop\Booggii\data'
            chunked_data_path = fr'C:\Users\GONY\Desktop\Booggii\processed_data\{window_minutes}min_{step_minutes}step_normalized_'

            os.makedirs(chunked_data_path, exist_ok=True)
            print('creating data')
            positive_data, negative_data = prepare_biomarkers_data(patients_dict, tags_path, data_path, time,
                                                                   normalize=True)
            negative_data = create_chunked_data(negative_data, patients_dict, window_minutes=window_minutes,
                                                step_minutes=step_minutes)
            positive_data = create_chunked_data(positive_data, patients_dict, window_minutes=window_minutes,
                                                step_minutes=step_minutes)
            positive_data.to_pickle(
                chunked_data_path + rf'\train_eval_positive_data_normalized_{window_minutes}min_{step_minutes}step.pkl')
            negative_data.to_pickle(
                chunked_data_path + rf'\train_eval_negative_data_normalized_{window_minutes}min_{step_minutes}step.pkl')

            eval_positive_data, eval_negative_data = prepare_biomarkers_data(eval_patients_dict, tags_path, data_path,
                                                                             time, normalize=True)
            eval_negative_data = create_chunked_data(eval_negative_data, eval_patients_dict,
                                                     window_minutes=window_minutes,
                                                     step_minutes=step_minutes)
            eval_positive_data = create_chunked_data(eval_positive_data, eval_patients_dict,
                                                     window_minutes=window_minutes,
                                                     step_minutes=step_minutes)
            eval_positive_data.to_pickle(
                chunked_data_path + rf'\test_positive_data_normalized_{window_minutes}min_{step_minutes}step.pkl')
            eval_negative_data.to_pickle(
                chunked_data_path + rf'\test_negative_data_normalized_{window_minutes}min_{step_minutes}step.pkl')
