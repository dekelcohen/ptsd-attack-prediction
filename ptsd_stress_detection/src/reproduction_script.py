import pandas as pd
import numpy as np

TAGS_FILE = r"D:\workdir\ptsd_stress_detection\refined_tags\TRAIL009_refined_tags_test.csv"

def debug_parsing():
    print(f"Loading {TAGS_FILE}...")
    try:
        df = pd.read_csv(TAGS_FILE)
        
        col_name = 'new_timestamp'
        if col_name not in df.columns:
            print(f"Column {col_name} not found!")
            return

        print("\n--- Inspecting values ---")
        # Get the failing row (index 5 in CSV is likely index 4 or 5 in DF depending on header)
        # CSV Line 6 is index 4 (0-based)
        # Line 6: 2025-08-20 06:11:10 IDT
        
        val_idx = 4
        if len(df) > val_idx:
            val = df.iloc[val_idx][col_name]
            print(f"Row {val_idx} raw value: '{val}'")
            print(f"Row {val_idx} repr: {repr(val)}")
            
            # Check for replace
            clean_val = str(val).replace(' IDT', '').replace(' IST', '')
            print(f"Cleaned value: '{clean_val}'")
            print(f"Match ' IDT'? {' IDT' in val}")
        
        print("\n--- Testing Vectorized Clean ---")
        ts_col = col_name
        if df[ts_col].dtype == object:
             clean_col = df[ts_col].astype(str).str.replace(' IDT', '', regex=False).str.replace(' IST', '', regex=False)
        else:
             clean_col = df[ts_col]
        
        print(f"Clean column sample:\n{clean_col.head(6)}")
        
        print("\n--- Testing pd.to_datetime ---")
        try:
            dt = pd.to_datetime(clean_col)
            print("Success!")
            print(dt.head(6))
        except Exception as e:
            print(f"pd.to_datetime failed: {e}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_parsing()
