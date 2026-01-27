import fastavro
import os

avro_file = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data\2025-03-30\DEV008-3YK3K15223\raw_data\v6\1-1-DEV008_1743319379.avro"

def inspect_sensors(filepath):
    try:
        with open(filepath, 'rb') as f:
            reader = fastavro.reader(f)
            for record in reader:
                raw = record.get('rawData', {})
                
                print("\n--- Temperature ---")
                temp = raw.get('temperature', {}) 
                # Print keys
                print(f"Keys: {list(temp.keys())}")
                if 'values' in temp:
                    print(f"Sample Value: {temp['values'][0] if temp['values'] else 'Empty'}")

                print("\n--- Accelerometer ---")
                acc = raw.get('accelerometer', {})
                print(f"Keys: {list(acc.keys())}")
                if 'x' in acc:
                    print(f"Sample X: {acc['x'][0] if acc['x'] else 'Empty'}")
                if 'values' in acc: # Check alternative structure
                    print(f"Sample Values: {acc['values'][0]}")
                    
                break
    except Exception as e:
        print(e)

if __name__ == "__main__":
    inspect_sensors(avro_file)
