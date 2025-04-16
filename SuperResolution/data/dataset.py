import h5py
import torch
from torch.utils.data import Dataset

class HDF5Dataset(Dataset):
    def __init__(self, hdf5_path):
        super(HDF5Dataset, self).__init__()
        self.h5_file_path = hdf5_path
        self.h5_file = None 
        self._open_file()

        if 'lr' not in self.h5_file or 'hr' not in self.h5_file:
             self.close()
             raise ValueError(f"HDF5 file {hdf5_path} is missing 'lr' or 'hr' dataset.")

        self.lr = self.h5_file['lr']
        self.hr = self.h5_file['hr']

        if self.lr.shape[0] != self.hr.shape[0]:
             self.close()
             raise ValueError(f"Mismatched number of patches in {hdf5_path}: LR={self.lr.shape[0]}, HR={self.hr.shape[0]}")

        self.length = self.lr.shape[0]
        if self.length == 0:
             self.close()
             raise ValueError(f"HDF5 file {hdf5_path} contains zero patches.")

    def _open_file(self):
         if self.h5_file is None:
             try:
                 self.h5_file = h5py.File(self.h5_file_path, 'r')
             except Exception as e:
                 raise IOError(f"Failed to open HDF5 file {self.h5_file_path}: {e}")

    def __getitem__(self, idx):
        if self.h5_file is None:
             self._open_file()
             if self.h5_file is None:
                  raise RuntimeError("HDF5 file is closed or failed to open. Cannot fetch item.")

        lr_patch = torch.from_numpy(self.lr[idx]).float()
        hr_patch = torch.from_numpy(self.hr[idx]).float()
        return lr_patch, hr_patch

    def __len__(self):
        return self.length

    def close(self):
        if self.h5_file:
            try:
                self.h5_file.close()
                print(f"DEBUG: Closed HDF5 file: {self.h5_file_path}") 
            except Exception as e:
                 print(f"Warning: Error closing HDF5 file {self.h5_file_path}: {e}")
            finally:
                 self.h5_file = None

    def __del__(self):
        self.close()