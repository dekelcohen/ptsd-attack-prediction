import pandas as pd

def load_participant_dates(excel_path):
    participant_dates = {}
    try:
        df = pd.read_excel(excel_path)
        # normalize column names to lower case for easier matching
        df.columns = df.columns.str.lower()
        
        # Identify columns
        id_col = next((c for c in df.columns if 'user_id' in c), None)
        start_col = next((c for c in df.columns if 'trial_starting_date' in c), None)
        end_col = next((c for c in df.columns if 'trial_ending_date' in c), None)
        
        if id_col and start_col and end_col:
            # Create dictionary: {User_ID: {'start_date': date, 'end_date': date}}
            for _, row in df.iterrows():
                uid = str(row[id_col]).strip()
                participant_dates[uid] = {
                    'start_date': row[start_col],
                    'end_date': row[end_col]
                }
            print(f"Loaded dates for {len(participant_dates)} participants from Excel.")
        else:
            print(f"Could not find required columns in Excel. Found: {df.columns.tolist()}")
            
    except Exception as e:
        print(f"Error loading Excel file: {e}")
    return participant_dates
