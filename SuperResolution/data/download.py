import os
import requests
import zipfile
import io
import time
from tkinter import messagebox 

def check_and_download_dataset(url, target_dir, dataset_name, status_callback):
    if os.path.isdir(target_dir) and any(fname.lower().endswith('.png') for fname in os.listdir(target_dir)):
        status_callback(f"Dataset '{dataset_name}' found at {target_dir}")
        return True

    parent_dir = os.path.dirname(target_dir)
    status_callback(f"Dataset '{dataset_name}' not found. Checking base directory '{parent_dir}'...")
    os.makedirs(parent_dir, exist_ok=True)

    #todo: move this trigger to GUI
    confirm = messagebox.askyesno("Download DIV2K Dataset",
                                  f"The required dataset '{dataset_name}' was not found.\n"
                                  f"It needs to be downloaded from:\n{url}\n\n"
                                  f"This is a large file (~3.5 GB) and download may take a while.\n"
                                  f"Extracted data will be stored in:\n{target_dir}\n\n"
                                  "Do you want to proceed with the download?")
    if not confirm:
        status_callback("Dataset download cancelled by user", warning=True)
        return False

    status_callback(f"Attempting download from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status() 

        total_size = int(response.headers.get('content-length', 0))
        total_size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        status_callback(f"Downloading {total_size_mb:.1f} MB...")

        zip_data_bytes = io.BytesIO()
        downloaded_size = 0
        chunk_size = 8192
        last_update_time = time.time()

        for data_chunk in response.iter_content(chunk_size=chunk_size):
            zip_data_bytes.write(data_chunk)
            downloaded_size += len(data_chunk)
            current_time = time.time()
            if current_time - last_update_time > 0.5:
                 if total_size > 0:
                     progress_perc = downloaded_size * 100 / total_size
                     downloaded_mb = downloaded_size / (1024 * 1024)
                     bar_len = 30
                     filled_len = int(bar_len * downloaded_size / total_size)
                     bar = '=' * filled_len + '-' * (bar_len - filled_len)
                     status_callback(f"Downloading: [{bar}] {downloaded_mb:.1f}/{total_size_mb:.1f} MB ({progress_perc:.1f}%)")
                 else:
                     downloaded_mb = downloaded_size / (1024 * 1024)
                     status_callback(f"Downloading: {downloaded_mb:.1f} MB...")
                 last_update_time = current_time

        status_callback("Download complete. Extracting (this may also take time)...")
        zip_data_bytes.seek(0)

        with zipfile.ZipFile(zip_data_bytes) as zf:
           for member in zf.infolist():
               member_path = os.path.abspath(os.path.join(parent_dir, member.filename))
               if not member_path.startswith(os.path.abspath(parent_dir)):
                   raise zipfile.BadZipFile(f"Attempted path traversal in zip file: {member.filename}")
           zf.extractall(path=parent_dir) 

        if os.path.isdir(target_dir) and any(fname.lower().endswith('.png') for fname in os.listdir(target_dir)):
             status_callback(f"Dataset successfully downloaded and extracted to {target_dir}")
             return True
        else:
             extracted_folders = [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]
             possible_target_in_subdir = os.path.join(parent_dir, dataset_name) 

             if dataset_name in extracted_folders and os.path.isdir(possible_target_in_subdir):
                 if not os.path.isdir(target_dir):
                      try:
                          os.rename(possible_target_in_subdir, target_dir)
                          status_callback(f"Renamed extracted folder to {target_dir}")
                          if os.path.isdir(target_dir) and any(fname.lower().endswith('.png') for fname in os.listdir(target_dir)):
                              return True
                      except OSError as ren_e:
                          status_callback(f"Could not rename extracted folder: {ren_e}", error=True)

                 elif os.path.isdir(possible_target_in_subdir) and any(fname.lower().endswith('.png') for fname in os.listdir(possible_target_in_subdir)):
                      status_callback(f"Dataset extracted to a subfolder: {possible_target_in_subdir}. Using this path.", warning=True)
                      return True 

             status_callback(f"Extraction failed or '{dataset_name}' PNG files not found after extraction in '{parent_dir}' or expected subfolders. Check contents.", error=True)
             return False

    except requests.exceptions.RequestException as e:
        status_callback(f"Download failed: {e}", error=True)
        return False
    except zipfile.BadZipFile as e:
        status_callback(f"Downloaded file is not a valid ZIP archive or contains invalid paths: {e}", error=True)
        return False
    except Exception as e:
        status_callback(f"An error occurred during dataset setup: {e}", error=True)
        import traceback
        traceback.print_exc()
        return False
