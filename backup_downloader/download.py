import gdown
import os
import sys

def download_backups():
    folder_url = "https://drive.google.com/drive/folders/1u3HgncinzsNB70NsSdIleWl2ol3jihSg?usp=sharing"
    output_dir = "/backups"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Starting download from folder: {folder_url}")
    try:
        # gdown.download_folder downloads all files from the folder
        # quiet=False to see progress
        # use_cookies=False to avoid issues in docker if not needed
        gdown.download_folder(url=folder_url, output=output_dir, quiet=False, remaining_ok=True)
        print("Download completed successfully.")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    download_backups()