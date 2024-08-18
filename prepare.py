import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Define the directories
input_dir = 'data'
train_dir = 'training_data'
eval_dir = 'eval_data'

# Create directories if they do not exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(eval_dir, exist_ok=True)

# Load all JSON files into a DataFrame with a source column
def load_json_files(directory):
    records = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        records.append({'text': item, 'source': filename})
                elif isinstance(data, dict):
                    for item in data.get('text', []):
                        records.append({'text': item, 'source': filename})
    return records

# Load data from the directory
records = load_json_files(input_dir)

# Convert to DataFrame
df = pd.DataFrame(records)

# Shuffle and split the data into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.1, random_state=42)

# Function to save DataFrame to JSON files, preserving source filenames
def save_df_to_json(df, directory, prefix):
    grouped = df.groupby('source')
    for source_file, group in grouped:
        num_files = 0
        rows_per_file = 1000  # Adjust if needed
        
        for start in range(0, len(group), rows_per_file):
            end = start + rows_per_file
            sub_df = group.iloc[start:end]
            num_files += 1
            filename = os.path.join(directory, f"{prefix}_{source_file}_{num_files}.json")
            sub_df[['text']].to_json(filename, orient='records', lines=True)
            print(f"Saved {filename}")

# Save training and evaluation data
save_df_to_json(train_df, train_dir, "train_data")
save_df_to_json(eval_df, eval_dir, "eval_data")

print("Data splitting and saving complete.")
