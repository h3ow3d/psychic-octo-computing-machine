import json
import os
from collections import defaultdict
from transformers import pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Check if a GPU is available and set the device accordingly
device = 0 if torch.cuda.is_available() else -1

# Initialize the NLP model with GPU support if available
nlp_model = pipeline('feature-extraction', model='bert-base-uncased', device=device)

# Load JSON files from a specified directory
def load_json_files(directory):
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as file:
                json_files.append(json.load(file))
    return json_files

# Generate semantic embeddings for a key
def get_key_embeddings(key):
    embedding = nlp_model(key)[0]
    return np.mean(embedding, axis=0)  # Averaging the embeddings for simplicity

# Compare keys using their semantic embeddings
def compare_keys(key1, key2):
    embedding1 = get_key_embeddings(key1)
    embedding2 = get_key_embeddings(key2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

# Merge keys based on semantic similarity
def merge_keys(key1, key2):
    similarity = compare_keys(key1, key2)
    if similarity > 0.8:  # Threshold for similarity
        return key1  # Logic to merge key names can be improved
    else:
        return key1, key2

# Extract KPIs and structure them
def extract_kpis(json_obj):
    kpis = []
    kpi_base_names = ['KPI', 'KPI 1', 'KPI 2', 'KPI 3']
    for i, base_name in enumerate(kpi_base_names):
        kpi_desc_key = f'{base_name} description' if i > 0 else 'KPI name and description'
        target_key = f'Target_{i-1}' if i > 0 else 'Good target'
        rating_key = f'Rating _{i-1}' if i > 0 else 'Rating'

        if kpi_desc_key in json_obj or base_name == 'KPI':  # Modified to handle both formats
            kpi = {
                "Name": base_name,
                "Description": json_obj.get(kpi_desc_key, json_obj.get("KPI name and description", "N/A")),
                "Rating": json_obj.get(rating_key.strip(), json_obj.get("Rating", "N/A")),
                "Target": str(json_obj.get(target_key, "N/A"))
            }
            kpis.append(kpi)
    return kpis

# Generate the structured schema from the JSON files
def generate_structured_schema(directory):
    json_files = load_json_files(directory)
    structured_data = []

    for json_obj in json_files:
        for item in json_obj:
            structured_item = {
                "Business Area": item.get("Business area or arms length body", "N/A"),
                "Comments": item.get("Comments", "N/A"),
                "Contract ID": item.get("Contract ID", "N/A"),  # Assumes 'Contract ID' if available
                "Contract Title": item.get("Contract title and description", item.get("Contract", "N/A")),
                "Department": item.get("Dept", item.get("Department", "N/A")),
                "Financial Year": item.get("Financial Year", "N/A"),
                "KPI": extract_kpis(item),
                "Months": ", ".join(item.get("Months", [])),
                "Quarter": item.get("Quarter", "N/A"),
                "Supplier": item.get("Supplier", "N/A"),
            }
            structured_data.append(structured_item)

    return structured_data

# Output the structured schema to a file
def output_structured_data(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

# Example usage
if __name__ == "__main__":
    # Define directories
    train_dir = 'training_data'
    eval_dir = 'eval_data'

    # Process and save structured data for training data
    print("Processing training data...")
    train_structured_schema = generate_structured_schema(train_dir)
    output_structured_data(train_structured_schema, 'structured_training_data.json')
    print("Saved structured training data to 'structured_training_data.json'")

    # Process and save structured data for evaluation data
    print("Processing evaluation data...")
    eval_structured_schema = generate_structured_schema(eval_dir)
    output_structured_data(eval_structured_schema, 'structured_eval_data.json')
    print("Saved structured evaluation data to 'structured_eval_data.json'")
