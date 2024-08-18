import pandas as pd
import json
import os

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                file_data = json.load(file)
                if isinstance(file_data, list):
                    data.extend(file_data)
                else:
                    data.append(file_data)
    return data

def preprocess_data(data):
    df = pd.json_normalize(data)
    df.fillna('Unknown', inplace=True)
    df.columns = df.columns.str.strip()
    return df

# Example usage
data = load_json_files('path_to_your_json_files')
df = preprocess_data(data)
df.to_csv('processed_data.csv', index=False)
