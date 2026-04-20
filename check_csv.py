import csv

with open("c:\\PYASL\\asl_data_real.csv", 'r') as f:
    reader = csv.reader(f)
    header = next(reader, None)
    print(f"Header length: {len(header)}")
    for i, row in enumerate(reader):
        feats = row[1:]
        if len(feats) not in [42, 46]:
            print(f"Row {i} has weird number of features: {len(feats)}")
