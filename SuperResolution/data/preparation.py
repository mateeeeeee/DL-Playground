import h5py
import time
import os
import glob
import numpy as np
from PIL import Image
from tkinter import messagebox

import config

def prepare_training_data_incremental(source_dir, hdf5_path, status_callback, scale,
                                    patch_size=config.DEFAULT_PATCH_SIZE,
                                    stride=config.DEFAULT_STRIDE,
                                    chunk_size=config.HDF5_CHUNK_SIZE):
    if scale < 2:
        status_callback(f"Invalid scale factor {scale} for preprocessing. Must be >= 2.", error=True)
        return False

    if os.path.exists(hdf5_path):
        status_callback(f"Preprocessed training data (x{scale}) found: {hdf5_path}")
        try:
            with h5py.File(hdf5_path, 'r') as f:
                if 'lr' in f and 'hr' in f and f['lr'].shape[0] > 0 and f['hr'].shape[0] > 0:
                    status_callback(f"HDF5 check ok: {f['lr'].shape[0]} patches.")
                    return True
                else:
                    status_callback(f"HDF5 file {hdf5_path} seems invalid or empty. Re-processing.", warning=True)
                    try:
                        os.remove(hdf5_path)
                        status_callback(f"Removed invalid HDF5 file: {hdf5_path}", warning=True)
                    except OSError as e:
                         status_callback(f"Error removing invalid HDF5 file {hdf5_path}: {e}", error=True)
                         return False
        except Exception as e:
            status_callback(f"Error reading existing HDF5 {hdf5_path}: {e}. Re-processing.", warning=True)
            if os.path.exists(hdf5_path):
                try:
                    os.remove(hdf5_path)
                    status_callback(f"Removed corrupted HDF5 file: {hdf5_path}", warning=True)
                except OSError as e:
                    status_callback(f"Error removing corrupted HDF5 file {hdf5_path}: {e}", error=True)
                    return False

    #todo: move this trigger to GUI
    confirm = messagebox.askyesno("Prepare Training Data",
                                  f"Need to preprocess images from:\n{source_dir}\n"
                                  f"into patches (HR {patch_size}x{patch_size}, stride {stride}, LR downscaled x{scale}) "
                                  f"and save to:\n{os.path.basename(hdf5_path)}\n\n"
                                  "This process can take a *very* long time (potentially hours) "
                                  "and requires significant disk space.\n\n"
                                  "Proceed with preprocessing?")
    if not confirm:
        status_callback("Preprocessing cancelled by user.", warning=True)
        return False

    status_callback(f"Starting x{scale} preprocessing for {source_dir}. This will take a while...")
    start_time = time.time()
    h5_file = None
    try:
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        h5_file = h5py.File(hdf5_path, 'w')
        hr_patch_size = patch_size 

        lr_shape = (0, 1, hr_patch_size, hr_patch_size)
        hr_shape = (0, 1, hr_patch_size, hr_patch_size)
        max_shape = (None, 1, hr_patch_size, hr_patch_size)
        chunk_shape = (chunk_size, 1, hr_patch_size, hr_patch_size)

        lr_dataset = h5_file.create_dataset('lr', lr_shape, maxshape=max_shape,
                                            dtype=np.float32, chunks=chunk_shape)
        hr_dataset = h5_file.create_dataset('hr', hr_shape, maxshape=max_shape,
                                            dtype=np.float32, chunks=chunk_shape)

        image_files = sorted(glob.glob(os.path.join(source_dir, '*.png'))) # Consider other formats?

        if not image_files:
            raise ValueError(f"No PNG image files found in {source_dir}. Please ensure the dataset was downloaded and extracted correctly.")

        total_patches = 0
        processed_images = 0
        current_patch_index = 0

        for i, img_path in enumerate(image_files):
            img_start_time = time.time()
            try:
                hr_img = Image.open(img_path).convert('RGB')

                if hr_img.width < hr_patch_size or hr_img.height < hr_patch_size:
                    status_callback(f"Skipping small image: {os.path.basename(img_path)} ({hr_img.width}x{hr_img.height})", warning=True)
                    continue

                hr_img_ycbcr = hr_img.convert('YCbCr')
                hr_y, _, _ = hr_img_ycbcr.split()

                hr_w, hr_h = hr_y.size
                lr_w, lr_h = max(1, hr_w // scale), max(1, hr_h // scale)

                lr_y = hr_y.resize((lr_w, lr_h), Image.Resampling.BICUBIC)
                lr_y_upscaled = lr_y.resize((hr_w, hr_h), Image.Resampling.BICUBIC)

                hr_y_np = np.array(hr_y, dtype=np.float32) / 255.0
                lr_y_upscaled_np = np.array(lr_y_upscaled, dtype=np.float32) / 255.0

                img_lr_patches = []
                img_hr_patches = []
                for r in range(0, hr_h - hr_patch_size + 1, stride):
                    for c in range(0, hr_w - hr_patch_size + 1, stride):
                        hr_patch = hr_y_np[r:r+hr_patch_size, c:c+hr_patch_size]
                        lr_patch = lr_y_upscaled_np[r:r+hr_patch_size, c:c+hr_patch_size]

                        img_lr_patches.append(lr_patch[np.newaxis, :, :])
                        img_hr_patches.append(hr_patch[np.newaxis, :, :])

                num_img_patches = len(img_lr_patches)
                if num_img_patches > 0:
                    new_size = current_patch_index + num_img_patches
                    lr_dataset.resize((new_size, 1, hr_patch_size, hr_patch_size))
                    hr_dataset.resize((new_size, 1, hr_patch_size, hr_patch_size))

                    lr_dataset[current_patch_index:new_size, ...] = np.array(img_lr_patches, dtype=np.float32)
                    hr_dataset[current_patch_index:new_size, ...] = np.array(img_hr_patches, dtype=np.float32)

                    total_patches += num_img_patches
                    current_patch_index = new_size

                processed_images += 1
                img_time = time.time() - img_start_time
                status_callback(f"Preprocessing x{scale}: Image {processed_images}/{len(image_files)} ({os.path.basename(img_path)}) -> {num_img_patches} patches ({img_time:.1f}s). Total: {total_patches}")

            except FileNotFoundError:
                 status_callback(f"Warning: Image file not found: {img_path}", warning=True)
                 continue
            except Exception as img_e:
                 print(f"Error processing image {img_path}: {img_e}")
                 import traceback
                 traceback.print_exc()
                 status_callback(f"Warning: Skipping {os.path.basename(img_path)} due to error: {img_e}", warning=True)
                 continue

        if total_patches == 0:
            if h5_file:
                try: h5_file.close()
                except: pass
                if os.path.exists(hdf5_path):
                    try: os.remove(hdf5_path)
                    except Exception as del_e: status_callback(f"Failed to delete empty HDF5 {hdf5_path}: {del_e}", error=True)
            raise ValueError("No patches were extracted. Cannot create training data. Check source images and parameters.")

        h5_file.close()
        h5_file = None
        end_time = time.time()
        status_callback(f"Training data preprocessing (x{scale}) complete. Saved {total_patches} patches to {os.path.basename(hdf5_path)}. Time: {(end_time - start_time)/60:.1f} min.")
        return True

    except Exception as e:
        status_callback(f"Error during preprocessing for x{scale}: {e}", error=True)
        import traceback
        traceback.print_exc()
        if h5_file:
            try: h5_file.close()
            except: pass 
        if os.path.exists(hdf5_path):
            status_callback(f"Deleting incomplete HDF5 file: {hdf5_path}", warning=True)
            try: os.remove(hdf5_path)
            except Exception as del_e: status_callback(f"Failed to delete {hdf5_path}: {del_e}", error=True)
        return False
