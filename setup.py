import os
import requests
import zipfile
from io import BytesIO

# URLs to download
zips = {
    "image_data.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    "objects.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
    "region_descriptions.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip",
    "attributes.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
    "relationships.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "attributes_synsets.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attribute_synsets.json.zip",
    "object_synsets.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/object_synsets.json.zip",
    "relationship_synsets.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationship_synsets.json.zip",
}

txts = {
    "object_alias.txt": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/object_alias.txt",
    "relationship_alias.txt": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationship_alias.txt",
}

# Create 'data' directory if it doesn't exist
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


def download_and_extract_zip(url, output_folder):
    """
    Downloads a ZIP file from a given URL and extracts it to the output folder.

    Parameters:
        url (str): The URL of the file to download.
        output_folder (str): The folder where the file should be extracted.
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        with zipfile.ZipFile(BytesIO(response.content)) as zfile:
            zfile.extractall(output_folder)
        print(f"Extracted to {output_folder}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
    except zipfile.BadZipFile as e:
        print(f"Error extracting file: {e}")


def download_txt(url, output_folder, filename):
    """
    Downloads a TXT file from a given URL and saves it to the output folder.

    Parameters:
        url (str): The URL of the file to download.
        output_folder (str): The folder where the file should be saved.
        filename (str): The name of the file to save.
    """
    try:
        print(f"Downloading {url}...")
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        file_path = os.path.join(output_folder, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)
        print(f"Saved {filename} to {output_folder}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")


# Download and extract ZIP files
for filename, url in zips.items():
    download_and_extract_zip(url, data_folder)

# Download TXT files
for filename, url in txts.items():
    download_txt(url, data_folder, filename)

print("Download and extraction complete.")
