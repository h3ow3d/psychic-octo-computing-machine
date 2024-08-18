import os
import pandas as pd
import pyexcel_ods3 as ods
import json
import re

# Directories
download_dir = "download"
output_dir = "data"

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to make column names unique
def make_columns_unique(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

# Function to parse the "Quarter" field
def parse_quarter(row):
    if pd.isna(row) or not isinstance(row, str) or not row.strip():
        return {"Financial Year": None, "Quarter": None, "Months": []}
    
    # Extract Financial Year, Quarter, and Months using regular expressions
    match = re.match(r'(FY\d{2}-\d{2})\s*(Q\d)\s*([A-Za-z\-]+)', row)
    
    if match:
        financial_year, quarter, months_str = match.groups()
        # Convert months_str to a list of months
        months = [month.strip() for month in months_str.split('-')]
        return {"Financial Year": financial_year, "Quarter": quarter, "Months": months}
    else:
        return {"Financial Year": None, "Quarter": None, "Months": []}

# Function to apply the parsing and add to the DataFrame
def parse_quarter_field(df):
    if 'Quarter' in df.columns:
        parsed_quarter = df['Quarter'].apply(parse_quarter).apply(pd.Series)
        df = df.drop(columns=['Quarter'])
        df = pd.concat([df, parsed_quarter], axis=1)
    return df

# Function to convert ODS to DataFrame
def ods_to_dataframe(file_path):
    data = ods.get_data(file_path)
    sheet_name = list(data.keys())[0]  # get the name of the first sheet
    df = pd.DataFrame(data[sheet_name])
    df.columns = df.iloc[0]
    df = df[1:]  # Skip the header row
    df = make_columns_unique(df)
    df = parse_quarter_field(df)
    return df

# Function to convert DataFrame to JSON
def dataframe_to_json(df):
    return df.to_json(orient='records', default_handler=str)  # Handle non-standard types

# Process each ODS file in the download directory
for file_name in os.listdir(download_dir):
    if file_name.endswith('.ods'):
        file_path = os.path.join(download_dir, file_name)
        
        df = ods_to_dataframe(file_path)
        json_data = dataframe_to_json(df)
        
        json_file_name = file_name.replace('.ods', '.json')
        json_file_path = os.path.join(output_dir, json_file_name)
        
        with open(json_file_path, 'w') as json_file:
            json_file.write(json_data)
        
        print(f"Converted {file_name} to {json_file_name} in {output_dir}")
