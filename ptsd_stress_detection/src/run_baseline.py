from pipeline import StressDetectionPipeline
import os

# DATA PATHS
# Note: Using TRAIL003 as matched set for now. 
# In production, need to iterate all participants and match their specific tag files.
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data" # Root to find avros
TAGS_FILE = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags\TRAIL003_valid_tags.csv"

def run_baseline():
    print("========================================")
    print("   BASELINE EXPERIMENT: XGBoost + Avro  ")
    print("========================================")
    
    pipeline = StressDetectionPipeline(use_model="xgboost")
    
    # Load invalid timestamps from aggregated CSV files (minutes with missing_value_reason)
    pipeline.load_invalid_timestamps_from_aggregated(DATA_DIR)
    
    # Run full pipeline
    # Note: run_full_pipeline now handles feature extraction + label alignment + training
    df_result = pipeline.run_full_pipeline(DATA_DIR, TAGS_FILE)
    
    if df_result is not None:
        print("\n[SUCCESS] Baseline Run Complete.")
        
        # Save Features for quick optimization iteration (caching)
        df_result.to_csv("baseline_features_labeled.csv", index=False)
        print("Features saved to 'baseline_features_labeled.csv'")
    else:
        print("\n[FAILURE] Pipeline failed.")

if __name__ == "__main__":
    run_baseline()
