import cProfile
import os
from math import ceil
from pathlib import Path
from pstats import Stats, SortKey

import pandas as pd
import pytz
from bokeh.layouts import column, gridplot
from bokeh.models import HoverTool, Div
from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose
from enum import Enum
import shutil

from src.avro_utils import update_csvs_from_new_avros

from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter

from preprocess_utils import PreprocessUtils
from avro_utils import generate_dataframes_from_avro


class Biomarker(Enum):
    Pr = "pulse-rate"
    Eda = "eda"  # Electrodermal Activity
    AccStd = "accelerometers-std"  # Accelerometer Magnitude Standard Deviation
    Prv = "prv"  # Pulse Rate Variability
    Met = "met"  # Metabolic Equivalent of Task
    Temp = "temperature"  # Skin Temperature in °C


biomarker_value_names = {
    Biomarker.Pr: "pulse_rate_bpm",
    Biomarker.Eda: "eda_scl_usiemens",
    Biomarker.AccStd: "accelerometers_std_g",
    Biomarker.Prv: "prv_rmssd_ms",
    Biomarker.Met: "met",
    Biomarker.Temp: "temperature_celsius"
}

# color_hex = "#FF0000"  # You'll likely see a red swatch here
# color_rgb = "rgb(0, 255, 0)"  # You'll likely see a green swatch here
# color_rgba = "rgba(0, 0, 255, 0.5)"  # You'll likely see a semi-transparent blue swatch
# color_name = "blue"  # You might see a blue swatch

biomarker_colors = {
    Biomarker.Pr: "crimson",
    Biomarker.Eda: "darkorange",
    Biomarker.AccStd: "aqua",
    Biomarker.Prv: "blue",
    Biomarker.Met: "green",
    Biomarker.Temp: "yellow",
}


def main():
    # pr = cProfile.Profile()
    # pr.enable()

    # user_id = "TRAIL002"
    # trial_starting_date = "2025-04-09 14:30:00"
    user_id = "TRAIL003"
    trial_starting_date = "2025-03-30 16:15:00"
    jerusalem_tz = pytz.timezone('Asia/Jerusalem')
    data_root_dir = Path("data/embrace_plus/")
    biomarker_names = [Biomarker.Pr, Biomarker.Eda, Biomarker.AccStd, Biomarker.Prv, Biomarker.Met, Biomarker.Temp]
    # biomarker_dfs, filtered_df = prepare_data(biomarker_names, data_root_dir, jerusalem_tz, trial_starting_date,
    #                                           user_id, override=False)
    prepare_data_from_avro(data_root_dir, jerusalem_tz, trial_starting_date,
                           user_id)
    # all_eda_df = pd.read_parquet(user_id + '/' + 'all_eda_df.parquet')
    # all_hr_df = pd.read_parquet(user_id + '/' + 'all_hr_df.parquet')
    # all_temp_df = pd.read_parquet(user_id + '/' + 'all_temp_df.parquet')

    # pr.disable()
    # Stats(pr).sort_stats(
    #     SortKey.CUMULATIVE).print_stats()
    # Stats(pr).sort_stats(
    #     SortKey.TIME).print_stats()
    # for biomarker_name, biomarker_df in biomarker_dfs.items():
    #     biomarker_df.plot.hist(bins=20, alpha=0.5)
    # merged_bio = pd.concat([x.set_index(['datetime']) for x in list(biomarker_dfs.values())], axis=1)

    # merged_bio_desc = merged_bio.describe(percentiles=[.05, .25, .5, .75, .95])
    # fig1 = plt.figure(1)
    # fig2 = plt.figure(2)
    # ax1 = fig1.subplots()
    # ax2 = fig2.subplots()
    # ax1.axis('off')
    # ax2.axis('off')
    # merged_bio_desc_table = pd.plotting.table(ax1, merged_bio_desc, loc='center', cellLoc='left')#, colWidths=list([.2, .2]))
    # merged_bio_desc_table.auto_set_font_size(False)
    # merged_bio_desc_table.set_fontsize(10)
    # merged_bio_corr = merged_bio.corr(method='spearman')
    # merged_bio_corr.style.background_gradient(cmap='coolwarm')
    # merged_bio_corr_table = pd.plotting.table(ax2, merged_bio_corr, loc='center', cellLoc='right')
    # merged_bio_corr_table.auto_set_font_size(False)
    # merged_bio_corr_table.set_fontsize(10)


    # plt.show()

    # visualize_data(biomarker_dfs, filtered_df, split=True)
    # visualize_events(biomarker_dfs, filtered_df)

    # systolic_peaks_df = pd.read_csv("data/embrace_plus/participant_data/2025-04-02/TRAIL003-3YK3K153QJ/raw_data/v6/1-1-TRAIL003_1743552982/systolic_peaks.csv", sep=',')
    # hr_per_sec_df = calculate_hr(systolic_peaks_df)


def calculate_hr(df: pd.DataFrame, timestamp_column: str = 'datetime') -> pd.DataFrame:
    # Convert the datetime column to datetime
    df['datetime'] = pd.to_datetime(df['datetime'], unit='us', utc=True).map(
        lambda x: x.tz_convert('Asia/Jerusalem'))
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

    # Create a new DataFrame with a row for every quarter of a second in the input DataFrame
    min_timestamp = df['rounded_timestamp'].min()
    max_timestamp = df['rounded_timestamp'].max()
    all_quarters = pd.date_range(start=min_timestamp, end=max_timestamp, freq='250ms')

    # Merge the new DataFrame with the result DataFrame to get heart rate per minute for every quarter of a second
    full_result_df = pd.DataFrame({'rounded_timestamp': all_quarters})
    full_result_df = full_result_df.merge(result_df, on='rounded_timestamp', how='left')

    # Interpolate the heart rate values to fill NaN values
    full_result_df['hr_per_minute'] = full_result_df['hr_per_minute'].interpolate()

    return full_result_df


def visualize_events(biomarker_dfs, filtered_df, time_delta=pd.Timedelta(hours=1)):
    event_plots = []
    for event_time in filtered_df['datetime']:
        filtered_biomarker_dfs = {}
        for biomarker_name, biomarker_df in biomarker_dfs.items():
            start_time = event_time - time_delta
            end_time = event_time + time_delta
            window_data = biomarker_df[
                (biomarker_df['datetime'] >= start_time) & (biomarker_df['datetime'] <= end_time)]

            if not window_data.empty:
                filtered_biomarker_dfs[biomarker_name] = window_data
            else:
                print(f"no data for event in {event_time} in biomarker {biomarker_name} with time delta of {time_delta}")

        event_plots.append(visualize_data(filtered_biomarker_dfs, event_time, split=False))
    show(gridplot(event_plots, ncols=4, sizing_mode="stretch_width"))


def visualize_data(biomarker_dfs, events, split=True):
    fig_big = figure(sizing_mode="stretch_width", x_axis_type='datetime', background_fill_color="WhiteSmoke")
    fig_big.xaxis.axis_label = 'Time'
    fig_big.xaxis.formatter = DatetimeTickFormatter(days="%d/%m",
                                                    hours="%H",
                                                    minutes="%H:%M")
    if split:
        fig_small = figure(sizing_mode="stretch_width", x_axis_type='datetime', background_fill_color="WhiteSmoke")
        fig_small.xaxis.axis_label = 'Time'
        fig_small.xaxis.formatter = DatetimeTickFormatter(days="%d/%m",
                                                          hours="%H",
                                                          minutes="%H:%M")
    for name, df in biomarker_dfs.items():
        proper_size_fig = fig_big
        if split and df[biomarker_value_names[name]].max() < 50:
            proper_size_fig = fig_small
        proper_size_fig.line(
            x=df["datetime"].dt.tz_localize(None),
            y=df[biomarker_value_names[name]],
            legend_label=name.value,
            color=biomarker_colors[name]
        )
    localized_events = events["datetime"].dt.tz_localize(None) if isinstance(events, DataFrame) else [events.tz_localize(None)]
    fig_big.vspan(
        x=localized_events,
        line_color="red",
        legend_label="events",
    )

    tooltips = [
        ('datetime', '@x{%Y-%m-%d %H:%M:%S}'),
        ('value', '@y'),
    ]
    fig_big.add_tools(HoverTool(tooltips=tooltips,
                                formatters={'@x': 'datetime'}))
    fig_big
    if split:
        fig_small.vspan(
            x=localized_events,
            line_color="red",
            legend_label="event",
        )
        fig_small.add_tools(HoverTool(tooltips=tooltips,
                                      formatters={'@x': 'datetime'}))
        layout = column(fig_big, fig_small, spacing=10, sizing_mode="stretch_both")
        show(layout)


    return column(fig_big, spacing=10, sizing_mode="stretch_both")


def prepare_data(biomarker_names, data_root_dir, jerusalem_tz, trial_starting_date,
                 user_id, override=False):
    participant_data_dir = data_root_dir.joinpath("participant_data")
    participant_processed_data_dir = data_root_dir.joinpath("participant_processed_data")
    trial_starting_datetime = pd.to_datetime(trial_starting_date, utc=True).tz_convert('Asia/Jerusalem')
    biomarker_dfs = {biomarker_name: DataFrame() for biomarker_name in biomarker_names}
    all_tags_df = DataFrame()
    all_hr_df = DataFrame()
    all_eda_df = DataFrame()
    all_temp_df = DataFrame()
    for date_dir in os.listdir(participant_data_dir):
        date_path = participant_data_dir.joinpath(date_dir)
        # if date_dir == "2025-04-01":
        #     break
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
                            df['datetime_utc'] = pd.to_datetime(df['timestamp_iso'],
                                                                utc=True)  # .map(lambda x: x.tz_convert('Asia/Jerusalem'))
                            df['datetime'] = df['datetime_utc'].dt.tz_convert(jerusalem_tz)
                            # df2 = df.set_index(['datetime'])
                            df2 = df.dropna(subset=[biomarker_value_names[biomarker_name]])
                            df2 = df2[["datetime", biomarker_value_names[biomarker_name]]]
                            biomarker_dfs[biomarker_name] = pd.concat([biomarker_dfs[biomarker_name], df2])
                raw_data_path = user_path.joinpath("raw_data/")
                csv_dirs = os.listdir(os.path.join(raw_data_path, "v6"))
                if override:
                    for csv_dir in csv_dirs:
                        if not csv_dir.endswith('avro'):
                            shutil.rmtree(os.path.join(raw_data_path, "v6", csv_dir))
                # update_csvs_from_new_avros(raw_data_path)
                for csv_dir in csv_dirs:
                    p = os.path.join(raw_data_path, "v6", csv_dirs[1])
                    generate_dataframes_from_avro(p)
                    if not csv_dir.endswith('avro'):
                        for filename in os.listdir(os.path.join(raw_data_path, "v6", csv_dir)):
                            file_path = os.path.join(raw_data_path, "v6", csv_dir, filename)
                            if filename == 'tags.csv':
                                try:
                                    tags_df = pd.read_csv(file_path, sep=',', header=None)
                                    tags_df["datetime"] = pd.to_datetime(tags_df[0], unit='us', utc=True).map(
                                        lambda x: x.tz_convert('Asia/Jerusalem'))
                                    all_tags_df = pd.concat([all_tags_df, tags_df])
                                except pd.errors.EmptyDataError:
                                    pass
                            elif filename == 'systolic_peaks.csv':
                                systolic_peaks_df = pd.read_csv(file_path, sep=',')
                                all_hr_df = pd.concat([all_hr_df, systolic_peaks_df])
                            elif filename == 'EDA.csv':
                                PreprocessUtils.read_sensor_files(sig_type='EDA',fmt='empatica_csv',root_path=cfg.DATA_FOLDER)
                                eda_df = pd.read_csv(file_path, sep=',')
                                all_eda_df = pd.concat([all_eda_df, eda_df])
                            elif filename == 'TEMP.csv':
                                temp_df = pd.read_csv(file_path, sep=',')
                                all_temp_df = pd.concat([all_temp_df, temp_df])

    all_hr_df = calculate_hr(all_hr_df)
    all_hr_df.to_parquet(user_processed_data_dir.joinpath("all_hr_df.parquet"), index=False)
    all_eda_df.to_parquet(user_processed_data_dir.joinpath("all_eda_df.parquet"), index=False)
    all_temp_df.to_parquet(user_processed_data_dir.joinpath("all_temp_df.parquet"), index=False)

    # false_tags_df = pd.read_csv("data/embrace_plus/participants_extra_data/false_tags/" + user_id + "_false_tags.csv",
    #                             sep=',', header=None)
    # false_tags_df["datetime"] = pd.to_datetime(false_tags_df[0])
    # all_tags_df['datetime'] = all_tags_df['datetime'].dt.floor('s')
    # filtered_df = all_tags_df[
    #     ~all_tags_df['datetime'].isin(false_tags_df['datetime']) & (all_tags_df['datetime'] >= trial_starting_datetime)]

    valid_tags_df = pd.read_csv("data/embrace_plus/participants_extra_data/valid_tags/" + user_id + "_valid_tags.csv",
                                sep=',')
    valid_tags_df["datetime"] = pd.to_datetime(valid_tags_df['timestamp'])

    return biomarker_dfs, valid_tags_df


def prepare_data_from_avro(data_root_dir, jerusalem_tz, trial_starting_date,
                 user_id):
    participant_data_dir = data_root_dir.joinpath("participant_data")
    participant_processed_data_dir = data_root_dir.joinpath("participant_processed_data")
    trial_starting_datetime = pd.to_datetime(trial_starting_date, utc=True).tz_convert('Asia/Jerusalem')
    all_sps_df = DataFrame()
    all_eda_df = DataFrame()
    all_temp_df = DataFrame()
    for date_dir in os.listdir(participant_data_dir):
        date_path = participant_data_dir.joinpath(date_dir)
        # if date_dir != "2025-04-21":
        #     continue
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
                        eda_df, temp_df, sps_df = generate_dataframes_from_avro(os.path.join(raw_data_path, "v6", csv_dir),
                                                                                jerusalem_tz)
                        if not eda_df.empty:
                            all_eda_df = pd.concat([all_eda_df, eda_df])
                        if not temp_df.empty:
                            all_temp_df = pd.concat([all_temp_df, temp_df])
                        if not sps_df.empty:
                            all_sps_df = pd.concat([all_sps_df, sps_df])

    all_sps_df = all_sps_df[(all_sps_df['datetime'] >= trial_starting_datetime)]
    all_eda_df = all_eda_df[(all_eda_df['datetime'] >= trial_starting_datetime)]
    all_temp_df = all_temp_df[(all_temp_df['datetime'] >= trial_starting_datetime)]

    all_hr_df = calculate_hr(all_sps_df)
    all_hr_df.to_parquet(user_processed_data_dir.joinpath("all_hr_df.parquet"), index=False)
    all_eda_df.to_parquet(user_processed_data_dir.joinpath("all_eda_df.parquet"), index=False)
    all_temp_df.to_parquet(user_processed_data_dir.joinpath("all_temp_df.parquet"), index=False)


if __name__ == '__main__':
    main()
