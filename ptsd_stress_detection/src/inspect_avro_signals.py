import fastavro
import os

avro_file = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data\2025-03-30\DEV008-3YK3K15223\raw_data\v6\1-1-DEV008_1743319379.avro"

def inspect_avro_signals(filepath):
    try:
        with open(filepath, 'rb') as f:
            reader = fastavro.reader(f)
            for record in reader:
                raw = record.get('rawData', {})
                
                print("\n--- EDA Structure ---")
                eda = raw.get('eda', {})
                for k, v in eda.items():
                    if isinstance(v, list):
                        print(f"  {k}: list len {len(v)}")
                    else:
                        print(f"  {k}: {v}")

                print("\n--- Systolic Peaks Structure ---")
                peaks = raw.get('systolicPeaks', {})
                for k, v in peaks.items():
                    if isinstance(v, list):
                        print(f"  {k}: list len {len(v)} (First: {v[0]})")
                    else:
                        print(f"  {k}: {v}")
                break
    except Exception as e:
        print(e)

if __name__ == "__main__":
    inspect_avro_signals(avro_file)
