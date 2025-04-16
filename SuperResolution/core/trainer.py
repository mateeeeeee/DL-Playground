import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import os

import config
from models.srcnn import SRCNN
from data.dataset import HDF5Dataset

def train_model(hdf5_path, save_path, train_scale, epochs, learning_rate, batch_size, batch_limit, device, status_callback):
    """ Trains the SRCNN model using the specified HDF5 dataset """
    train_dataset = None
    dataloader = None
    try:
        status_callback(f"Training x{train_scale}: Initializing model, optimizer...")
        model = SRCNN(num_channels=1).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        status_callback(f"Training x{train_scale}: Loading data from {os.path.basename(hdf5_path)}...")

        try:
            train_dataset = HDF5Dataset(hdf5_path)
            status_callback(f"Dataset contains {len(train_dataset)} patches.")
        except (IOError, ValueError) as e:
            status_callback(f"Error loading dataset {hdf5_path}: {e}", error=True)
            return False

        if len(train_dataset) == 0:
            status_callback(f"Dataset at {hdf5_path} is empty. Cannot train.", error=True)
            if train_dataset: train_dataset.close() 
            return False

        num_workers = config.NUM_DATALOADER_WORKERS
        persistent_workers = num_workers > 0 
        pin_memory = device.type == 'cuda'

        status_callback(f"Using {num_workers} workers for DataLoader.")

        try:
            dataloader = DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory,
                                    persistent_workers=persistent_workers,
                                    drop_last=True if len(train_dataset) % batch_size != 0 else False)
        except Exception as dl_e:
            status_callback(f"Error creating DataLoader (try setting num_workers=0): {dl_e}", error=True)
            if train_dataset: train_dataset.close()
            return False

        model.train()

        total_batches_per_epoch = len(dataloader)
        if total_batches_per_epoch == 0:
             status_callback(f"DataLoader is empty. Check batch size ({batch_size}) vs dataset size ({len(train_dataset)}).", error=True)
             if train_dataset: train_dataset.close()
             return False

        batches_to_run_per_epoch = total_batches_per_epoch
        if batch_limit > 0:
             batches_to_run_per_epoch = min(total_batches_per_epoch, batch_limit)

        status_callback(f"Training x{train_scale} started: {epochs} epochs, {batches_to_run_per_epoch} batches/epoch (limit: {batch_limit if batch_limit > 0 else 'None'}).")
        start_time_total = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            processed_batches_epoch = 0

            for i, (lr_batch, hr_batch) in enumerate(dataloader):

                if batch_limit > 0 and i >= batch_limit:
                    status_callback(f"Training x{train_scale}: Reached batch limit ({batch_limit}) for epoch {epoch+1}.")
                    break

                try:
                    lr_batch = lr_batch.to(device, non_blocking=pin_memory)
                    hr_batch = hr_batch.to(device, non_blocking=pin_memory)
                except Exception as move_e:
                     status_callback(f"Error moving batch {i} to device: {move_e}", error=True)
                     continue 

                outputs = model(lr_batch)
                loss = criterion(outputs, hr_batch)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                processed_batches_epoch += 1

                report_interval = max(1, batches_to_run_per_epoch // 20)
                if (i + 1) % report_interval == 0 or (i + 1) == batches_to_run_per_epoch:
                    batch_loss = loss.item()
                    progress_perc = (i + 1) * 100 / batches_to_run_per_epoch
                    time_so_far_epoch = time.time() - epoch_start_time
                    eta_epoch_s = 0
                    if (i + 1) > 0 and time_so_far_epoch > 0:
                        eta_epoch_s = (time_so_far_epoch / (i + 1)) * (batches_to_run_per_epoch - (i + 1))
                    eta_epoch_m = eta_epoch_s / 60
                    progress_msg = (
                        f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{batches_to_run_per_epoch}] "
                        f"({progress_perc:.1f}%), Loss: {batch_loss:.6f}, ETA Epoch: {eta_epoch_m:.1f}m"
                    )
                    status_callback(f"Training x{train_scale}: {progress_msg}")

            if processed_batches_epoch > 0:
                avg_epoch_loss = epoch_loss / processed_batches_epoch
                epoch_time = time.time() - epoch_start_time
                status_callback(f"Training x{train_scale}: Epoch {epoch+1}/{epochs} finished. Avg Loss: {avg_epoch_loss:.6f}, Time: {epoch_time:.2f}s")
            else:
                 status_callback(f"Training x{train_scale}: Epoch {epoch+1}/{epochs} skipped (no batches processed).", warning=True)

        total_train_time = time.time() - start_time_total
        status_callback(f"Training x{train_scale}: Saving final model to {os.path.basename(save_path)}...")
        try:
            torch.save(model.state_dict(), save_path)
            status_callback(f"Training x{train_scale} complete. Model saved. Total time: {total_train_time/60:.1f} minutes.")
            if train_dataset:
                train_dataset.close()
            return True
        except Exception as save_e:
            status_callback(f"Error saving model to {save_path}: {save_e}", error=True)
            if train_dataset: train_dataset.close()
            return False

    except Exception as e:
        status_callback(f"An unexpected error occurred during training x{train_scale}: {e}", error=True)
        import traceback
        traceback.print_exc() 
        return False
    finally:
        if train_dataset:
            status_callback(f"DEBUG: Closing dataset in finally block...")
            train_dataset.close()
