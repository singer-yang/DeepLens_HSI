import os

import requests
from tqdm import tqdm


def download_kaist_dataset(
    urls_file="datasets/kaist.txt", output_dir="./datasets/kaist/"
):
    """
    Download the Kaist dataset EXR files from the provided URLs.

    Args:
        urls_file (str): Path to the file containing URLs.
        output_dir (str): Directory where downloaded files will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read URLs from file
    with open(urls_file, "r") as f:
        urls = [line.strip() for line in f if line.strip()]

    print(f"Found {len(urls)} files to download.")

    # Download each file
    for url in tqdm(urls, desc="Downloading files"):
        # Extract filename from URL
        filename = url.split("/")[-1]
        output_path = os.path.join(output_dir, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"File {filename} already exists, skipping.")
            continue

        try:
            # Download file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Get file size for progress reporting
            total_size = int(response.headers.get("content-length", 0))

            # Write to file
            with open(output_path, "wb") as out_file:
                if total_size == 0:  # No content length header
                    out_file.write(response.content)
                else:
                    # Use tqdm for progress bar within each file download
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=filename,
                        leave=False,
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                out_file.write(chunk)
                                pbar.update(len(chunk))

            print(f"Downloaded {filename}")

        except Exception as e:
            print(f"Error downloading {url}: {e}")

    print("Download complete!")


if __name__ == "__main__":
    download_kaist_dataset()
