from pipeline import StressDetectionPipeline
import pandas as pd
import numpy as np
import os

def create_mock_aggregated_data():
    """
    Generates mock files behaving like the real 1-min aggregated data.
    """
    os.makedirs("mock_data/dev1/digital_biomarkers/aggregated_per_minute", exist_ok=True)
    
    # Generate 60 mins of data (1 sample per min)
    timestamps_ms = np.arange(1743292800000, 1743292800000 + 60*60*1000, 60*1000)
    
    # EDA
    df_eda = pd.DataFrame({
        'timestamp_unix': timestamps_ms,
        'eda_scl_usiemens': np.random.uniform(0.1, 5.0, size=len(timestamps_ms)),
        'timestamp_iso': '2025-...'
    })
    eda_path = "mock_data/dev1/digital_biomarkers/aggregated_per_minute/1-1-DEV_eda.csv"
    df_eda.to_csv(eda_path, index=False)
    
    # HR
    df_hr = pd.DataFrame({
        'timestamp_unix': timestamps_ms,
        'pulse_rate_bpm': np.random.uniform(60, 100, size=len(timestamps_ms))
    })
    hr_path = "mock_data/dev1/digital_biomarkers/aggregated_per_minute/1-1-DEV_pulse-rate.csv"
    df_hr.to_csv(hr_path, index=False)
    
    return "mock_data"

def run_test():
    root = create_mock_aggregated_data()
    
    pipeline = StressDetectionPipeline(use_model="xgboost")
    
    print("\n--- Running on Mock Aggregated Data ---")
    df = pipeline.run_on_all_data(root)
    
    if df is not None:
        print("\n--- Result Head ---")
        print(df.head())
        
        # Verify features
        expected = ['eda_roll_mean', 'hr_slope']
        missing = [c for c in expected if c not in df.columns]
        
        if not missing:
            print("\n[SUCCESS] Aggregated features computed.")
        else:
            print(f"\n[FAILURE] Missing: {missing}")
            
        # Dummy Train
        df['label'] = np.random.randint(0, 2, size=len(df))
        pipeline.classifier.train(df.drop(columns=['label', 'timestamp']), df['label'])
    else:
        print("[FAILURE] No data returned.")

    # Cleanup (optional)
    import shutil
    try:
        shutil.rmtree("mock_data")
    except:
        pass

if __name__ == "__main__":
    run_test()
