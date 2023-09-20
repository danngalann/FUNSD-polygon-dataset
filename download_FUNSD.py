import os
import zipfile
import requests

# Constants
URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
FOLDER = "datasets/"
ZIP_FILE_NAME = "dataset.zip"
EXTRACTION_SUBFOLDER = "FUNSD"

def download_file(url, destination):
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def unzip_file(zip_path, output_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

def main():
    # Create folder if it doesn't exist
    if not os.path.exists(FOLDER):
        os.makedirs(FOLDER)

    # Specify the folder to extract to
    extraction_folder = os.path.join(FOLDER, EXTRACTION_SUBFOLDER)
    if not os.path.exists(extraction_folder):
        os.makedirs(extraction_folder)

    # Download the zip file
    zip_destination = os.path.join(FOLDER, ZIP_FILE_NAME)
    download_file(URL, zip_destination)

    # Unzip the file
    unzip_file(zip_destination, extraction_folder)

    # Optionally, remove the zip file after extraction
    os.remove(zip_destination)

if __name__ == "__main__":
    main()
