import pandas as pd
import numpy as np

TAGS_FILE = r"D:\workdir\ptsd_stress_detection\refined_tags\TRAIL009_refined_tags_test.csv"

def validate():
    print(f"Validating {TAGS_FILE}...")
    df = pd.read_csv(TAGS_FILE)
    
    # Parse timestamps
    df['original_ts'] = pd.to_datetime(df['timestamp'].str.replace(" IDT", "").str.replace(" IST", "")).dt.tz_localize('Israel', ambiguous='NaT', nonexistent='shift_forward')
    df['refined_ts'] = pd.to_datetime(df['new_timestamp'], format='ISO8601')
    
    # Calculate shift in minutes
    # Shift = refined - original
    df['shift_seconds'] = (df['refined_ts'] - df['original_ts']).dt.total_seconds()
    df['shift_minutes'] = df['shift_seconds'] / 60
    
    # Stats
    print("\n=== Refinement Statistics ===")
    print(f"Total Tags: {len(df)}")
    
    shifted = df[abs(df['shift_seconds']) > 1] # Tolerance 1 sec
    print(f"Tags Shifted: {len(shifted)} ({len(shifted)/len(df)*100:.1f}%)")
    
    if len(shifted) > 0:
        print(f"Min Shift: {shifted['shift_minutes'].min():.2f} min")
        print(f"Max Shift: {shifted['shift_minutes'].max():.2f} min")
        print(f"Mean Shift: {shifted['shift_minutes'].mean():.2f} min")
        
        # Check constraints
        # Constraint: max 6h before (-360 min) or 30m after (+30 min)
        # Shift should be >= -360 and <= 30
        
        violations = df[(df['shift_minutes'] < -360) | (df['shift_minutes'] > 30)]
        
        if not violations.empty:
            print(f"\n[WARNING] {len(violations)} tags violate constraints!")
            print(violations[['timestamp', 'new_timestamp', 'shift_minutes']])
        else:
            print("\n[SUCCESS] All tags are within [-6h, +30m] constraint.")
            
        # Histogram
        print("\nShift Distribution (deciles):")
        print(shifted['shift_minutes'].quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
        
if __name__ == "__main__":
    validate()
