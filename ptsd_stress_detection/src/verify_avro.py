from pipeline import StressDetectionPipeline
import os

# Target the directory containing the real Avro file we inspected
real_data_dir = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data\2025-03-30\DEV008-3YK3K15223\raw_data\v6"

def run_real_avro_test():
    pipeline = StressDetectionPipeline(use_model="xgboost")
    
    print(f"\n--- Running Pipeline on REAL Avro Data ---")
    print(f"Target: {real_data_dir}")
    
    # Run pipeline
    df_features = pipeline.run_on_all_data(real_data_dir)
    
    if df_features is not None and not df_features.empty:
        print("\n[SUCCESS] Features Extracted from Avro!")
        print(f"Total Windows: {len(df_features)}")
        print(df_features.head())
        
        # Check specific RAW features (from Deconvolution)
        expected = ['eda_tonic_mean', 'eda_scr_count', 'hr_mean', 'temp_mean', 'acc_std']
        present = [c for c in expected if c in df_features.columns]
        print(f"Verified SOTA Features: {present}")
        
    else:
        print("\n[FAILURE] No features extracted. Check logical constraints (min window size, data duration).")

if __name__ == "__main__":
    run_real_avro_test()
