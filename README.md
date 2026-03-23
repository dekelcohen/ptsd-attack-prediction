# PTSD Stress Detection Project

This project implements a machine learning pipeline to detect stress events using physiological data from the Empatica EmbracePlus watch (EDA, Heart Rate, Temperature, Accelerometer).

## **Current Performance (5 Participants)**
- **F1 Score**: **0.54** (Baseline: 0.37 with 5 participants, 0.50 with 2)
- **Precision**: 0.60
- **Recall**: 0.49
- **Optimization**: Uses **Personalized Z-Score Normalization** and a high decision threshold (0.80).

---

## **End-to-End Workflow**

### **1. Raw Data Ingestion**
- **Source**: Empatica AVRO files (1-second resolution or raw sampling).
- **Streams**:
  - `eda`: Electrodermal Activity (4Hz)
  - `systolicPeaks`: Heart Rate / IBI
  - `accelerometer`: Movement (for activity filtering)
  - `temperature`: Skin temperature

### **2. Label Refinement (`label_refiner.py`)**
User-reported labels are often inaccurate (delayed or forgotten). We refine them logic:
- **Cluster Walk-Back**: Finds the physiological "onset" of stress preceding the user tag.
- **Origin Scoring**: Prioritizes "Watch+Remote" tags (High Confidence) over "App-only" (Retrospective).
- **Severity Filter**: Removes tags with severity < 1.
- **Output**: `_refined_tags_v2.csv` containing physiologically aligned timestamps.

### **3. Preprocessing & Feature Extraction (`pipeline.py`)**
- **Cleaning**:
  - Despiking (sigma=3)
  - Elliptic Filtering (1Hz low-pass for EDA)
- **Feature Engineering** (10-minute sliding windows):
  - **EDA**: Phasic (peaks), Tonic (background), AUC.
  - **HRV**: RMSSD, pNN50, SDNN (derived from IBI).
  - **Context**: Accelerometer Magnitude (`acc_mean`).
  - **Cross-Modal**: EDA/HR ratios.

### **4. Personalized Normalization (Crucial Step)**
*Solved the "Baseline Divergence" problem (e.g., User A HR=60 vs User B HR=90).*
- Features are **Z-scored (Standardized) per participant** before training.
- Formula: $z = (x - \mu_{user}) / \sigma_{user}$
- This allows the model to learn "Relative Arousal" rather than absolute values.

### **5. Model Training (`run_enhanced_cv.py`)**
- **Algorithm**: LightGBM (Gradient Boosting).
- **Strategy**: 
  - **Combined Training**: Pooled data from all participants.
  - **Stratified CV**: Ensures stress events are balanced across folds.
  - **Threshold Tuning**: Custom loop searches for optimal decision boundary (found best at **0.80**).

---

## **How to Run**

### **1. Refine Labels (One-time)**
```bash
python ptsd_stress_detection/src/label_refiner.py
```

### **2. Train Model**
```bash
python ptsd_stress_detection/src/run_enhanced_cv.py lightgbm --combined
```

### **3. Analyze Data Distributions**
```bash
python ptsd_stress_detection/src/analyze_feature_distributions.py
```

## **Key insights**
1. **Normalization is non-negotiable**: Without it, F1 dropped to <0.10 for some users due to baseline shifts.
2. **Context matters**: High physical activity mimics stress. While a strict filter removed some true positives, the model implicitly learns to ignore extremely high activity when provided with `acc_mean`.
3. **Data Quality**: "Watch-based" tags align 90% better with physiology than "App-based" retrospective tags.
