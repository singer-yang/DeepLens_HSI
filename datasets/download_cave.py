import os
import zipfile

import requests
from tqdm import tqdm


def download_cave_dataset(
    url="https://cave.cs.columbia.edu/old/databases/multispectral/zip/complete_ms_data.zip",
    output_dir="./datasets/cave/",
    zip_path=None,
):
    """
    Download the CAVE multispectral dataset and extract it.

    Args:
        url (str): URL to the dataset zip file.
        output_dir (str): Directory where extracted files will be saved.
        zip_path (str): Path where the zip file will be temporarily saved.
                        If None, it will be saved in the output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set default zip path if not provided
    if zip_path is None:
        zip_filename = url.split("/")[-1]
        zip_path = os.path.join(output_dir, zip_filename)

    # Check if the zip file was already downloaded
    if os.path.exists(zip_path):
        print(f"Zip file already exists at {zip_path}")
    else:
        print(f"Downloading dataset from {url}")
        try:
            # Get file size for progress reporting
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            total_size = int(response.headers.get("content-length", 0))

            # Download with progress bar
            with open(zip_path, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            print(f"Downloaded to {zip_path}")
        except Exception as e:
            print(f"Error downloading the file: {e}")
            return

    # Extract the zip file
    print(f"Extracting files to {output_dir}")
    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Get total number of files for progress reporting
            total_files = len(zip_ref.infolist())

            # Extract with progress bar
            for i, file in enumerate(zip_ref.infolist()):
                zip_ref.extract(file, output_dir)
                print(f"Extracted: {i + 1}/{total_files} files", end="\r")
        print("\nExtraction complete!")

        # Remove the zip file after extraction
        os.remove(zip_path)
        print(f"Removed zip file {zip_path}")

    except Exception as e:
        print(f"Error extracting the zip file: {e}")


if __name__ == "__main__":
    download_cave_dataset()
