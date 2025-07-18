from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytz
from avro.datafile import DataFileReader
from avro.io import DatumReader
import json
import csv
import os
import fastavro
from pandas import DataFrame


def generate_csvs_from_avro(avro_file_path: Path):
    output_dir = avro_file_path.with_suffix('')
    output_dir.mkdir()

    # Read Avro file
    reader = DataFileReader(open(avro_file_path, "rb"), DatumReader())
    schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))
    data = next(reader)

    # Uncomment the below 2 lines to print the Avro schema
    # print(schema)
    # print(" ")

    # Export sensors data to csv files
    # Accelerometer
    acc = data["rawData"]["accelerometer"]
    timestamp = acc["timestampStart"] / 1e6

    # Convert ADC counts in g
    delta_physical = acc["imuParams"]["physicalMax"] - acc["imuParams"]["physicalMin"]
    delta_digital = acc["imuParams"]["digitalMax"] - acc["imuParams"]["digitalMin"]
    x_g = [val * delta_physical / delta_digital for val in acc["x"]]
    y_g = [val * delta_physical / delta_digital for val in acc["y"]]
    z_g = [val * delta_physical / delta_digital for val in acc["z"]]
    with open(os.path.join(output_dir, 'ACC.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, timestamp, timestamp])
        writer.writerow([acc["samplingFrequency"], acc["samplingFrequency"], acc["samplingFrequency"]])
        writer.writerows([[x, y, z] for x, y, z in zip(x_g, y_g, z_g)])

    # Gyroscope
    gyro = data["rawData"]["gyroscope"]
    timestamp = gyro["timestampStart"] / 1e6
    # Convert ADC counts in dps (degree per second)
    delta_physical = gyro["imuParams"]["physicalMax"] - gyro["imuParams"]["physicalMin"]
    delta_digital = gyro["imuParams"]["digitalMax"] - gyro["imuParams"]["digitalMin"]
    x_dps = [val * delta_physical / delta_digital for val in gyro["x"]]
    y_dps = [val * delta_physical / delta_digital for val in gyro["y"]]
    z_dps = [val * delta_physical / delta_digital for val in gyro["z"]]
    with open(os.path.join(output_dir, 'GYRO.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, timestamp, timestamp])
        writer.writerow([gyro["samplingFrequency"], gyro["samplingFrequency"], gyro["samplingFrequency"]])
        writer.writerows([[x, y, z] for x, y, z in zip(x_dps, y_dps, z_dps)])

    # Eda
    eda = data["rawData"]["eda"]
    with open(os.path.join(output_dir, 'EDA.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([eda["timestampStart"] / 1e6])
        writer.writerow([eda["samplingFrequency"]])
        writer.writerows([[value] for value in eda["values"]])

    # Temperature
    tmp = data["rawData"]["temperature"]
    timestamp = tmp["timestampStart"] / 1e6
    with open(os.path.join(output_dir, 'TEMP.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp])
        writer.writerow([tmp["samplingFrequency"]])
        writer.writerows([[value] for value in tmp["values"]])

    # Tags
    tags = data["rawData"]["tags"]
    with open(os.path.join(output_dir, 'tags.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[tag] for tag in tags["tagsTimeMicros"]])

    # BVP
    bvp = data["rawData"]["bvp"]
    timestamp = bvp["timestampStart"] / 1e6
    with open(os.path.join(output_dir, 'BVP.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp])
        writer.writerow([bvp["samplingFrequency"]])
        writer.writerows([[value] for value in bvp["values"]])

    # Systolic peaks (IBI)
    # TODO: Make it possible to use it with old IBI csv format
    sps = data["rawData"]["systolicPeaks"]
    with open(os.path.join(output_dir, 'systolic_peaks.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["systolic_peak_timestamp"])
        writer.writerows([[sp / 1e6] for sp in sps["peaksTimeNanos"]])

    # Steps
    steps = data["rawData"]["steps"]
    timestamp = steps["timestampStart"] / 1e6
    with open(os.path.join(output_dir, 'STEPS.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp])
        writer.writerow([steps["samplingFrequency"]])
        writer.writerows([[value] for value in steps["values"]])


def generate_dataframes_from_avro(avro_file_path: Path, local_tz=pytz.timezone('Asia/Jerusalem')):
    with open(avro_file_path, 'rb') as avro_file:
        reader = fastavro.reader(avro_file)

        data = next(reader)
        utc_tz = pytz.timezone('UTC')
        # Eda
        eda = data["rawData"]["eda"]
        sampling_rate = eda["samplingFrequency"]
        if sampling_rate == 0:
            eda_df = DataFrame()
        else:
            start_unix_timestamp = eda["timestampStart"] / 1e6
            start_dt_utc = utc_tz.localize(datetime.utcfromtimestamp(start_unix_timestamp))
            num_values = len(eda["values"])
            time_intervals = np.arange(0, num_values) / sampling_rate
            datetimes = start_dt_utc + pd.to_timedelta(time_intervals, unit='s')
            eda_df = pd.DataFrame({'datetime': datetimes, 'value': eda["values"]})
            if not eda_df.empty:
                eda_df['datetime'] = eda_df['datetime'].dt.tz_convert(local_tz)

        # Temperature
        temp = data["rawData"]["temperature"]
        sampling_rate = temp["samplingFrequency"]
        if sampling_rate == 0:
            temp_df = DataFrame()
        else:
            start_unix_timestamp = temp["timestampStart"] / 1e6
            start_dt_utc = utc_tz.localize(datetime.utcfromtimestamp(start_unix_timestamp))
            num_values = len(temp["values"])
            time_intervals = np.arange(0, num_values) / sampling_rate
            datetimes = start_dt_utc + pd.to_timedelta(time_intervals, unit='s')
            temp_df = pd.DataFrame({'datetime': datetimes, 'value': temp["values"]})
            if not temp_df.empty:
                temp_df['datetime'] = temp_df['datetime'].dt.tz_convert(local_tz)

        # Systolic peaks (IBI)
        sps = data["rawData"]["systolicPeaks"]
        sps_df = pd.DataFrame({'datetime': [utc_tz.localize(datetime.utcfromtimestamp(sp / 1e9)) for sp in sps["peaksTimeNanos"]]})
        if not sps_df.empty:
            sps_df['datetime'] = sps_df['datetime'].dt.tz_convert(local_tz)

    return eda_df, temp_df, sps_df


def update_db_event(participant, timestamp):
    api_url = 'https://r4jlflfk41.execute-api.eu-west-1.amazonaws.com/Dev/events'
    payload = {
        "patientId": participant,
        "deviceId": participant,
        "timestamp": str(timestamp),
        "location": {
            "lat": 0.0,
            "long": 0.0
        },
        "eventType": "other",
        "activity": "other",
        "severity": 4,
        "origin": "watch"
    }
    headers = {
        'Content-Type': 'application/json',
    }
    print(payload)
    # response = requests.patch(api_url, headers=headers, json=payload)
    # print(response.status_code)
    # print(response.text)

def update_csvs_from_new_avros(avro_root_dir):
    for user_dir in os.listdir(avro_root_dir):
        filenames = os.listdir(os.path.join(avro_root_dir, user_dir))
        for filename in filenames:
            if filename.endswith('avro'):
                if not os.path.isdir(os.path.join(avro_root_dir, user_dir, filename.removesuffix('.avro'))):
                    generate_csvs_from_avro(Path(os.path.join(avro_root_dir, user_dir, filename)))



# avro_root_dir = Path("data/embrace_plus/participants_data/007/2025-02-18/007-3YK3K15223/raw_data")
# update_csvs_from_new_avros(avro_root_dir)

'v2/566/1/1/participant_data/2025-02-18/007-3YK3K15223/raw_data/v6/1-1-007_1739855451.avro'
'v2/566/1/1/participant_data/2025-02-18/007-3YK3K15223/raw_data/v6/1-1-007_1739855451.avro'