import os
import pandas as pd

def test_write():
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Created/Checked {cache_dir}. Absolute: {os.path.abspath(cache_dir)}")
    
    path = os.path.join(cache_dir, "test_file.pkl")
    df = pd.DataFrame({"a": [1, 2, 3]})
    try:
        df.to_pickle(path)
        print(f"Successfully wrote pickle to {path}")
    except Exception as e:
        print(f"Failed to write: {e}")
        
    files = os.listdir(cache_dir)
    print(f"Files in cache: {files}")

if __name__ == "__main__":
    test_write()
