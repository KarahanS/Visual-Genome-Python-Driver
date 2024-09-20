import os
import requests
import zipfile
from io import BytesIO

# URLs to download
urls = {
    "image_data.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    "objects.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/objects.json.zip",
    "region_descriptions.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip",
    "attributes.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attributes.json.zip",
    "relationships.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/relationships.json.zip",
    "attributes_synsets.json.zip": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/attribute_synsets.json.zip",
}

# Create 'data' directory if it doesn't exist
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


def download_and_extract(url, output_folder):
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


# Download and extract each file
for filename, url in urls.items():
    download_and_extract(url, data_folder)

print("Download and extraction complete.")
