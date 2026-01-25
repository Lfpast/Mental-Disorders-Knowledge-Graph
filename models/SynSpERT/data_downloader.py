import os
import urllib.request
import zipfile
import ssl
import shutil

# Define the URL and target directory
FILE_URL = "https://zenodo.org/records/10960357/files/MDIEC.zip?download=1"
TARGET_DIR = os.path.join("..", "InputsAndOutputs", "input", "dataset")
ZIP_FILE_PATH = os.path.join(TARGET_DIR, "MDIEC.zip")

def download_file(url, target_path):
    print(f"Downloading data from {url}...")
    
    # Create an unverified SSL context to avoid potential SSL certificate errors
    ssl_context = ssl._create_unverified_context()
    
    try:
        with urllib.request.urlopen(url, context=ssl_context) as response, open(target_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        print(f"Download completed: {target_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        exit(1)

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction completed.")
    except Exception as e:
        print(f"Error extracting file: {e}")
        exit(1)

def main():
    # 1. Create target directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        print(f"Creating directory: {TARGET_DIR}")
        os.makedirs(TARGET_DIR)
    
    # 2. Download the file
    download_file(FILE_URL, ZIP_FILE_PATH)
    
    # Detect the root folder inside the zip before unzipping
    extracted_root_folder = None
    try:
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            if file_list:
                # Check if all files are inside a single top-level directory
                top_dir = file_list[0].split('/')[0]
                is_single_top_dir = all(f.startswith(top_dir + '/') or f == top_dir or f == top_dir + '/' for f in file_list)
                if is_single_top_dir:
                    extracted_root_folder = top_dir
    except Exception as e:
        print(f"Warning: Could not determine zip structure: {e}")

    # 3. Unzip the file
    unzip_file(ZIP_FILE_PATH, TARGET_DIR)
    
    # 4. Remove the zip file
    print(f"Removing zip file: {ZIP_FILE_PATH}")
    try:
        os.remove(ZIP_FILE_PATH)
    except OSError as e:
        print(f"Error removing zip file: {e}")

    # 5. Move files from extracted root folder to TARGET_DIR and delete root folder
    if extracted_root_folder:
        extracted_folder_path = os.path.join(TARGET_DIR, extracted_root_folder)
        if os.path.exists(extracted_folder_path) and os.path.isdir(extracted_folder_path):
            print(f"Moving contents from {extracted_folder_path} to {TARGET_DIR}...")
            for item in os.listdir(extracted_folder_path):
                source = os.path.join(extracted_folder_path, item)
                destination = os.path.join(TARGET_DIR, item)
                
                # Handle existing files in destination
                if os.path.exists(destination):
                    if os.path.isdir(destination):
                        shutil.rmtree(destination)
                    else:
                        os.remove(destination)
                
                shutil.move(source, destination)
            
            print(f"Removing empty directory: {extracted_folder_path}")
            os.rmdir(extracted_folder_path)
    else:
        print("No single top-level folder detected, skipping flatten step.")

if __name__ == "__main__":
    main()
