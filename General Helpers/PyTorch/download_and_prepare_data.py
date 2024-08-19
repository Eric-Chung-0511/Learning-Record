
import os
import zipfile
from pathlib import Path
import requests
import sys

def download_and_prepare_data(data_url, data_folder_name):
    # Setup path to data folder
    data_path = Path("data/")
    dataset_path = data_path / data_folder_name

    # If the dataset folder doesn't exist, download it and prepare it...
    if dataset_path.is_dir():
        print(f"{dataset_path} directory exists.")
    else:
        print(f"Did not find {dataset_path} directory, creating one...")
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Download the dataset
        zip_file_path = data_path / f"{data_folder_name}.zip"
        with open(zip_file_path, "wb") as f:
            print(f"Downloading {data_folder_name} data...")
            request = requests.get(data_url)
            f.write(request.content)

        # Unzip the dataset
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            print(f"Unzipping {data_folder_name} data...")
            zip_ref.extractall(dataset_path)

        # Remove zip file
        os.remove(zip_file_path)
        print(f"Removed zip file: {zip_file_path}")

if __name__ == "__main__":
    # Check if the script received the correct number of arguments
    if len(sys.argv) != 3:
        print("Usage: python download_and_prepare_data.py <data_url> <data_folder_name>")
        sys.exit(1)
    
    # Pass the arguments to the function
    DATA_URL = sys.argv[1]
    DATA_FOLDER_NAME = sys.argv[2]

    download_and_prepare_data(DATA_URL, DATA_FOLDER_NAME)
