import os
from enum import Enum
from pathlib import Path

import pandas as pd
import pytz
from bokeh.layouts import column, gridplot
from bokeh.models import DatetimeTickFormatter
from bokeh.models import HoverTool
from bokeh.plotting import figure, show
from matplotlib import pyplot as plt
from pandas import DataFrame


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

# color_hex = "#FF0000"  # red
# color_rgb = "rgb(0, 255, 0)"  # green
# color_rgba = "rgba(0, 0, 255, 0.5)"  # semi-transparent blue
# color_name = "blue"  # blue

biomarker_colors = {
    Biomarker.Pr: "crimson",
    Biomarker.Eda: "darkorange",
    Biomarker.AccStd: "aqua",
    Biomarker.Prv: "blue",
    Biomarker.Met: "green",
    Biomarker.Temp: "yellow",
}


def main():

    user_id = "TRAIL005"
    trial_starting_date = "2025-05-14 11:56:00"
    jerusalem_tz = pytz.timezone('Asia/Jerusalem')
    data_root_dir = Path("data/embrace_plus/")

    biomarker_names = [Biomarker.Pr, Biomarker.Eda, Biomarker.AccStd, Biomarker.Prv, Biomarker.Met, Biomarker.Temp]
    biomarker_dfs, filtered_df = prepare_data(biomarker_names, data_root_dir, jerusalem_tz, trial_starting_date,
                                              user_id, override=False)

    visualize_data(biomarker_dfs, filtered_df, split=True)
    visualize_events(biomarker_dfs, filtered_df)
    visualize_statistics(biomarker_dfs)


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
                print(
                    f"no data for event in {event_time} in biomarker {biomarker_name} with time delta of {time_delta}")

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
    localized_events = events["datetime"].dt.tz_localize(None) if isinstance(events, DataFrame) else [
        events.tz_localize(None)]
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

    valid_tags_df = pd.read_csv(
        "data/embrace_plus/participants_extra_data/valid_tags/" + user_id + "_valid_tags.csv", sep=',')
    if not valid_tags_df.empty:
        # Notice: manually replace in valid tags csv the " IDT" with "+03:00"
        valid_tags_df["datetime"] = pd.to_datetime(valid_tags_df['timestamp']).dt.tz_convert(
            jerusalem_tz)  # Convert to Jerusalem time

    return biomarker_dfs, valid_tags_df




if __name__ == '__main__':
    main()
