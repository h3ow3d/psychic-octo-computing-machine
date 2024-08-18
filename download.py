import requests
import re
import os

# List of URLs pointing to ODS files
urls = [
    "https://assets.publishing.service.gov.uk/media/669f64ab0808eaf43b50d489/20240722-JAN-FEB-MAR_Final_Data_For_Publication_Excluding_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/6627c658d29479e036a7e6a1/20240423-OCT-NOV-DEC_Final_Data_For_Publication_Excluding_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/660d32f0758315001a4a4977/20240403-JUL-AUG-SEP_Final_Data_For_Publication_Excluding_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/6538daf680884d0013f71a73/20231024-APR-JUN_2023_Final_Data_For_Publication_Excluding_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/64be3d301e10bf000e17ccb0/20230724-JAN-MAR_2023_Final_Data_For_Publication_Excluding_MOJ-O.ods",
    "https://assets.publishing.service.gov.uk/media/645bb1e2479612000cc29372/20230503-OCT-DEC_Final_Data_For_Publication_excluding_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/63ea2658e90e0706cca645fd/20230120-JUL-SEP_2022_Final_Data_For_Publication_Excluding_MOJ_.ods",
    "https://assets.publishing.service.gov.uk/media/63cab7d2e90e07071a597487/20230120-JUL-SEP_2022_Final_Data_For_Publication_Excluding_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/6380d037e90e0723368ec705/20221125-Apr-Jun_2022_Final_Data_for_Publication_excl_MOJ.ods",
    "https://assets.publishing.service.gov.uk/media/62e24440e90e07143d519304/2022-07-28_Jan-Mar_2022_Final_Data_For_Publication_Excluding_MoJ.ods",
    "https://assets.publishing.service.gov.uk/media/62695d60e90e0746cec75aee/2022-04-28_Oct-Dec_2021_Final_Data_For_Publication_Excluding_MoJ.ods",
    "https://assets.publishing.service.gov.uk/media/61f27a76d3bf7f78e0ff6632/2022-01-27_Jul-Sep_2021_Final_Data_For_Publication_Excluding_MoJ.ods",
    "https://assets.publishing.service.gov.uk/media/617a8027e90e07197648904f/2021-10-28_Apr-Jun_2021_Final_Data_For_Publication_Excluding_MoJ.ods",
    "https://assets.publishing.service.gov.uk/media/61028ccfd3bf7f044ee523ab/2021-07-29_Jan-Mar_2021_Final_Data_For_Publication_Excluding_MoJ.ods",
    "https://assets.publishing.service.gov.uk/media/60953494e90e0735772d9bbe/2021-05-07_Oct-Dec_2020_Final_Data_for_Publication_excluding_MoJ.ods",
    "https://assets.publishing.service.gov.uk/media/601bf28ad3bf7f70bf58f68c/2021-02-04_Main_KPI_Dataset_for_Publication_Jul-Sept_2020.ods",
    "https://assets.publishing.service.gov.uk/media/601bfc2ee90e0711c8c3d469/2021-02-04_Main_KPI_Dataset_for_Publication_Apr-Jun_2020.ods",
    "https://assets.publishing.service.gov.uk/media/5fbfa50de90e077edae2e11b/2020-11-26_Main_KPI_Dataset_for_Publication_Jan-Mar_2020.ods"
]

# Directory to save downloaded files
download_dir = "download"

# Create the directory if it does not exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Function to extract date from URL
def extract_date_from_url(url):
    # This regex pattern will try to match various date formats in the URL
    date_patterns = [
        r'/(\d{8})-',        # Matches YYYYMMDD at the beginning
        r'/(\d{4}-\d{2}-\d{2})_'  # Matches YYYY-MM-DD in the middle
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, url)
        if match:
            # Replace any dashes with nothing to standardize format to YYYYMMDD
            return match.group(1).replace('-', '')
    
    # If no date is found, return a fallback name or raise an error
    return "unknown_date"

# Download and save each ODS file with date in filename
for url in urls:
    response = requests.get(url)
    
    # Extract date from URL
    file_date = extract_date_from_url(url)
    
    # Create a filename with the date and specify the directory
    filename = os.path.join(download_dir, f"kpi_data_{file_date}.ods")
    
    with open(filename, 'wb') as f:
        f.write(response.content)
    
    print(f"Downloaded {filename}")
