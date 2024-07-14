import json
from pathlib import Path

import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

eda_file_path = Path("data/embrace_plus/2024-05-28/0010-3YK3K15223/digital_biomarkers/aggregated_per_minute/1-1-0010_2024-05-28_eda.csv")
pr_file_path = Path("data/embrace_plus/2024-05-28/0010-3YK3K15223/digital_biomarkers/aggregated_per_minute/1-1-0010_2024-05-28_pulse-rate.csv")
eda_df = pd.read_csv(eda_file_path, header=0, sep=',')
pr_df = pd.read_csv(pr_file_path, header=0, sep=',')

avro_file_path = Path("data/embrace_plus/2024-05-28/0010-3YK3K15223/raw_data/v6/1-1-0010_1716925835.avro")
reader = DataFileReader(open(avro_file_path, "rb"), DatumReader())
schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
data = next(reader)

from bokeh.plotting import figure, show
from bokeh.models import DatetimeTickFormatter, Range1d, LinearAxis

fig = figure(sizing_mode="stretch_both")
fig.xaxis.axis_label = 'Time'
fig.xaxis.formatter=DatetimeTickFormatter(days="%m/%d",
hours="%H",
minutes="%H:%M")

# Define 1st LHS y-axis
fig.yaxis.axis_label = 'EDA [μS]'
fig.y_range = Range1d(start=0, end=30)

# # Create 2nd LHS y-axis
# fig.extra_y_ranges['temp'] = Range1d(start=0, end=50)
# fig.add_layout(LinearAxis(y_range_name='temp', axis_label='Temperature [°C]'), 'left')

# Create 1st RHS y-axis
fig.extra_y_ranges['pr'] = Range1d(start=50, end=200)
fig.add_layout(LinearAxis(y_range_name='pr', axis_label='PR [BPM]'), 'right')

# # Create 2nd RHS y-axis
# fig.extra_y_ranges['gflow'] = Range1d(start=0, end=50)
# fig.add_layout(LinearAxis(y_range_name='gflow', axis_label='Gas Flowrate [MMscf/day]'), 'right')

eda_times = pd.to_datetime(eda_df.timestamp_iso)

fig.line(
    x=eda_times,
    y=eda_df.eda_scl_usiemens,
    legend_label='EDA',
    color='blue'
)
pr_times = pd.to_datetime(pr_df.timestamp_iso)

fig.line(
    x=pr_times,
    y=pr_df.pulse_rate_bpm,
    legend_label='PR',
    y_range_name='pr',
    color='red'
)

import neurokit2 as nk
# nk.hrv(pr_df., sampling_rate=100, show=True)

#
# fig.line(
#     x = [0,1,2,3,4,5],
#     y = [10000,10100,10000,10150,9990,10000],
#     legend = 'Liquid Flowrate',
#     y_range_name = 'lflow',
#     color = 'orange'
# )
#
#
# fig.line(
#     x = [0,1,2,3,4,5],
#     y = [35,37,40,41,40,36],
#     legend = 'Gas Flowrate',
#     y_range_name = 'gflow',
#     color = 'green'
# )

show(fig)











# from avro.datafile import DataFileReader
# from avro.io import DatumReader
#
# reader = DataFileReader(open("data/embrace_plus/sample/1-1-0000000001_1705568933.avro", "rb"), DatumReader())
# for user in reader:
#     print(user)
# reader.close()

# def prepare_data(avro_data):
#     prepare_eda(eda_data)
#
# def prepare_eda(eda_data):

# from bokeh.plotting import figure, show
# from bokeh.models import LinearAxis, Range1d, RangeTool, DatetimeTickFormatter
# from datetime import datetime
#
# color = ['red', 'green', 'magenta', 'black']
# p = figure(sizing_mode="stretch_both")
# # times = df_eda.timestamp.apply(lambda x: x.time())
# times = pd.to_datetime(df_eda.timestamp)
# # p = figure(tools="xpan", toolbar_location=None,
# #            x_axis_location="above",
# #            background_fill_color="#efefef", sizing_mode="stretch_both")
# p.line(times, df_eda['signal'], color='blue')
# p.xaxis.formatter=DatetimeTickFormatter(days="%m/%d",
# hours="%H",
# minutes="%H:%M")
#
# # p.line('date', 'close', source=source)
# p.yaxis.axis_label = 'Price'
#
# select = figure(title="Drag the middle and edges of the selection box to change the range above",
#                 height=130, width=800, y_range=p.y_range,
#                 x_axis_type="datetime", y_axis_type=None,
#                 tools="", toolbar_location=None, background_fill_color="#efefef")
#
# range_tool = RangeTool(x_range=p.x_range)
# range_tool.overlay.fill_color = "navy"
# range_tool.overlay.fill_alpha = 0.2
#
# select.line(df_eda.timestamp.apply(lambda x: x.time().isoformat()), df_eda['signal'], color='blue')
# select.ygrid.grid_line_color = None
# select.add_tools(range_tool)
# select.toolbar.active_multi = range_tool
#
# show(column(p, select))
# for i, c in enumerate(color, start=1):
#     name = f'extra_range_{i}'
#     lable = f'extra range {i}'
#     p.extra_y_ranges[name] = Range1d(start=0, end=10*i)
#
#     p.add_layout(LinearAxis(axis_label=lable, y_range_name=name), 'left')
#     p.line(data_x, data_y, color=c, y_range_name=name)
# show(p)


# import plotly
# import cufflinks as cf
# cf.go_offline()
# reduce_by = len(df_eda) / 50000 + 1
# # df_eda_avg = df_eda.groupby(np.arange(len(df_eda))//reduce_by).mean()
# # df_eda_avg.iplot(x='timestamp', y='signal', kind='line')
# df_eda.describe(include='all')
# import plotly.graph_objects as go
# fig = go.Figure(
#     data=df_eda,
#     layout_title_text="A Figure Displayed with fig.show()"
# )
# fig.show(renderer="iframe")
########### ECG Polar test
# windows_fts,features_cols = featut.gen_feature_windows_for_type(
#     preut=preut, sig_type='ECG', fmt='polar_csv',df_tags=df_tags, window_start=0, window_end=5*60)
