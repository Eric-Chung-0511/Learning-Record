import os
import zipfile

from pathlib import Path

import requests

def download_data(source: str,
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.

    Returns:
        pathlib.Path to downloaded data.

    Example usage:
        download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                      destination="pizza_steak_sushi")
    """
    # setup path to data folder
    data_path = Path('data/')
    image_path = data_path /destination

    if image_path.is_dir():
        print(f'[INFO] {image_path} directory already exist, skipping download.')
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # download the target data
        target_file = Path(source).name
        with open(data_path / target_file, 'wb') as f:
            request = requests.get(source) # download link
            print(f"[INFO] Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip the file
        with zipfile.ZipFile(data_path / target_file, 'r') as zip_ref:
             print(f"[INFO] Unzipping {target_file} data...")
             zip_ref.extractall(image_path)

        # remove .zip file if needed
        if remove_source:
            os.remove(data_path / target_file)

    return image_path

# Usage
# image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
#                            destination="pizza_steak_sushi")
# image_path
