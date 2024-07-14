from pathlib import Path

from avro.datafile import DataFileReader
from avro.io import DatumReader
import json
import csv
import os


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
        writer.writerows([[sp] for sp in sps["peaksTimeNanos"]])

    # Steps
    steps = data["rawData"]["steps"]
    timestamp = steps["timestampStart"] / 1e6
    with open(os.path.join(output_dir, 'STEPS.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp])
        writer.writerow([steps["samplingFrequency"]])
        writer.writerows([[value] for value in steps["values"]])


def update_csvs_from_new_avros(avro_root_dir):
    for user_dir in os.listdir(avro_root_dir):
        filenames = os.listdir(os.path.join(avro_root_dir, user_dir))
        for filename in filenames:
            if filename.endswith('avro'):
                if not os.path.isdir(os.path.join(avro_root_dir, user_dir, filename.removesuffix('.avro'))):
                    generate_csvs_from_avro(Path(os.path.join(avro_root_dir, user_dir, filename)))


avro_root_dir = Path("data/embrace_plus/2024-05-28/0010-3YK3K15223/raw_data")
update_csvs_from_new_avros(avro_root_dir)
