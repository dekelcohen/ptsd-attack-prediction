import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pipeline import StressDetectionPipeline

# Config
DATA_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participant_data"
TAGS_DIR = r"D:\workdir\ptsd-attack-prediction\data\embrace_plus\participants_extra_data\valid_tags\auto_modified_tags"
OUTPUT_CSV = "all_participants_results.csv"
OUTPUT_IMG = "f1_scores_all_participants.png"

def main():
    print("=== STARTING BATCH PROCESSING FOR ALL PARTICIPANTS ===")
    
    # 1. Discover Participants with Modified Tags
    tag_files = glob.glob(os.path.join(TAGS_DIR, "*_valid_tags_modified.csv"))
    print(f"Found {len(tag_files)} tag files.")
    
    results = []

    for tags_file in tag_files:
        basename = os.path.basename(tags_file)
        participant_id = basename.split("_")[0] # e.g. TRAIL001
        
        print(f"\n--- Processing Participant: {participant_id} ---")
        
        try:
            # Initialize Pipeline
            pipeline = StressDetectionPipeline(use_model="xgboost")
            
            # Load invalid timestamps from aggregated CSV files (minutes with missing_value_reason)
            pipeline.load_invalid_timestamps_from_aggregated(DATA_DIR)
            
            # Find relevant avro files
            all_files = pipeline.find_avro_files(DATA_DIR)
            participant_files = [f for f in all_files if participant_id in f]
            
            if len(participant_files) < 50:
                print(f"Skipping {participant_id}: Too few files ({len(participant_files)})")
                continue
                
            print(f"Found {len(participant_files)} avro files for {participant_id}")
            
            # Extract Features (cached)
            features_df = pipeline.process_files(participant_files)
            
            if features_df is None or features_df.empty:
                print(f"Skipping {participant_id}: No features extracted.")
                continue
                
            # Load Tags
            tags_df = pipeline.load_labels(tags_file)
            if tags_df is None or tags_df.empty:
                print(f"Skipping {participant_id}: No tags loaded.")
                continue
                
            # Align Labels
            df_final = pipeline.align_labels(features_df, tags_df)
            
            # Check class balance
            n_pos = df_final['label'].sum()
            n_neg = len(df_final) - n_pos
            print(f"Class Balance: 0={n_neg}, 1={n_pos}")
            
            if n_pos < 5:
                print(f"Skipping {participant_id}: Too few positive samples ({n_pos})")
                continue
            
            # Prepare X, y
            drop_cols = ['label', 'start_time', 'end_time', 'source_file', 'timestamp']
            feature_cols = [c for c in df_final.columns if c not in drop_cols]
            
            X = df_final[feature_cols]
            y = df_final['label']
            X = X.fillna(0)
            
            # Run Evaluation
            # Note: train_and_evaluate_cv_threshold now returns (best_f1, best_prec, best_rec)
            best_f1, best_prec, best_rec = pipeline.classifier.train_and_evaluate_cv_threshold(X, y)
            
            results.append({
                "Participant": participant_id,
                "F1_Score": best_f1,
                "Precision": best_prec,
                "Recall": best_rec,
                "Windows": len(df_final),
                "Stress_Events": n_pos
            })
            
            print(f"--> {participant_id} Result: F1={best_f1:.4f}")
            
        except Exception as e:
            print(f"!!! Error processing {participant_id}: {e}")
            continue

    # 2. Save Results
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("F1_Score", ascending=False)
        results_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved results to {OUTPUT_CSV}")
        
        # 3. Print Summary Table
        print("\n=== FINAL RESULTS SUMMARY ===")
        print(results_df.to_markdown(index=False, floatfmt=".4f"))
        
        # 4. Generate Infographic
        generate_plot(results_df)
    else:
        print("\nNo results generated.")

def generate_plot(df):
    try:
        plt.figure(figsize=(12, 6))
        sns.set_style("whitegrid")
        
        # Color bar based on F1 threshold
        colors = ['green' if x >= 0.7 else 'blue' if x >= 0.5 else 'red' for x in df['F1_Score']]
        
        ax = sns.barplot(x="Participant", y="F1_Score", data=df, palette=colors)
        
        plt.title("Stress Detection Performance by Participant (F1 Score)", fontsize=16)
        plt.xlabel("Participant ID")
        plt.ylabel("Best F1 Score")
        plt.axhline(0.7, color='red', linestyle='--', label='Target (0.7)')
        plt.legend()
        plt.xticks(rotation=45)
        plt.ylim(0, 1.0)
        
        # Add labels on top of bars
        for i, v in enumerate(df['F1_Score']):
            ax.text(i, v + 0.01, f"{v:.2f}", ha='center', fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(OUTPUT_IMG)
        print(f"Generated infographic: {OUTPUT_IMG}")
        
    except Exception as e:
        print(f"Failed to generate plot: {e}")

if __name__ == "__main__":
    main()
