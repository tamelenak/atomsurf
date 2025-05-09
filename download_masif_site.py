import os
import requests
import tarfile
from tqdm import tqdm
import argparse
import sys

def download_file(url, filename):
    """
    Download a file with progress bar
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(filename, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
        
        # Verify the file was downloaded correctly
        if os.path.getsize(filename) == 0:
            raise Exception("Downloaded file is empty")
            
    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
        if os.path.exists(filename):
            os.remove(filename)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Download MaSIF-site dataset')
    parser.add_argument('--output_dir', type=str, default='masif_site_data',
                        help='Directory to save the downloaded data')
    parser.add_argument('--dataset', type=str, choices=['site', 'ligand'], default='site',
                        help='Which dataset to download: site (3.2GB) or ligand (10.7GB)')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # MaSIF-site dataset URLs
    urls = {
        'site': "https://zenodo.org/record/2625420/files/masif_site_masif_search_pdbs_and_ply_files.tar.gz",
        'ligand': "https://zenodo.org/record/2625420/files/masif_ligand_pdbs_and_ply_files.tar.gz"
    }
    
    url = urls[args.dataset]
    output_file = os.path.join(args.output_dir, os.path.basename(url))

    print(f"Downloading MaSIF {args.dataset} dataset to {output_file}...")
    print(f"URL: {url}")
    download_file(url, output_file)

    print("Extracting files...")
    try:
        with tarfile.open(output_file, 'r:gz') as tar:
            tar.extractall(path=args.output_dir)
    except tarfile.ReadError as e:
        print(f"Error extracting file: {e}")
        print("The downloaded file might be corrupted. Please try downloading again.")
        if os.path.exists(output_file):
            os.remove(output_file)
        sys.exit(1)

    # Remove the tar file after extraction
    os.remove(output_file)
    print("Download and extraction complete!")

if __name__ == "__main__":
    main() 