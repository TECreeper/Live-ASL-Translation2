import csv
import os
import shutil

def migrate_file(filepath):
    if not os.path.exists(filepath):
        print(f"{filepath} does not exist.")
        return
        
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None: return
        
        # Checking if already migrated
        if len(header) >= 47:
            print(f"{filepath} is already migrated or has unexpected number of features: {len(header)}")
            return
            
        header += ['Feature42', 'Feature43', 'Feature44', 'Feature45']
        
        for row in reader:
            if len(row) > 0:
                row += ['0.0', '0.0', '0.0', '0.0']
                rows.append(row)
                
    temp_path = filepath + ".tmp"
    with open(temp_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        
    try:
        os.replace(temp_path, filepath)
        print(f"Migrated {filepath} successfully.")
    except Exception as e:
        print(f"Failed to replace {filepath}: {e}")

if __name__ == "__main__":
    migrate_file("c:\\PYASL\\asl_data_real.csv")
    migrate_file("c:\\PYASL\\hand_detection_data.csv")
