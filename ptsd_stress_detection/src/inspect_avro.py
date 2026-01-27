import fastavro
import json
import os

# Target a specific avro file found earlier
avro_file = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data\2025-03-30\DEV008-3YK3K15223\raw_data\v6\1-1-DEV008_1743319379.avro"

def inspect_avro(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    try:
        with open(filepath, 'rb') as f:
            reader = fastavro.reader(f)
            print("--- Schema ---")
            print(json.dumps(reader.writer_schema, indent=2))
            
            print("\n--- First 2 Records ---")
            for i, record in enumerate(reader):
                print(record)
                if i >= 1: break
    except Exception as e:
        print(f"Error reading avro: {e}")

if __name__ == "__main__":
    inspect_avro(avro_file)
