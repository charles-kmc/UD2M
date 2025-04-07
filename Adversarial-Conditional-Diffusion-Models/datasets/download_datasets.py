import requests
import zipfile
import os

# download CT dataset
def download_large_zip_shared(shared_link, local_path):
    """Download a large ZIP file using a Dropbox shared link."""
    direct_link = shared_link.replace("?dl=0", "?dl=1")  
    with requests.get(direct_link, stream=True) as response:
        response.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024): 
                f.write(chunk)
    print(f"Downloaded to {local_path}")

def unzip_file(zip_path, extract_to):
    """Extract a ZIP file and delete it after extraction."""
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted files to {extract_to}")
    os.remove(zip_path)
    print(f"Deleted {zip_path} after extraction")


if __name__=="__main__":
    # Example Usage
    shared_link = "https://www.dropbox.com/scl/fo/ybc183siiymibubwqu00n/AJZo9oNUHZoVxJK9zy5S76Q/training_set.zip?rlkey=l0lz28rinw8b5ozylf1x1tgn0&dl=1"
    local_zip_path = "/users/cmk2000/sharedscratch/Datasets/ct_training_set.zip"
    extract_folder = "/users/cmk2000/sharedscratch/Datasets/training_CT_data"

    # Download the large ZIP file
    download_large_zip_shared(shared_link, local_zip_path)

    # Unzip it
    unzip_file(local_zip_path, extract_folder)
