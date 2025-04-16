import os
import torch

DEFAULT_MODEL_FILENAME_PATTERN = "srcnn_model_div2k_x{}.pth"

DEFAULT_TRAIN_UPSCALE_FACTOR = 2
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_BATCH_LIMIT = 0  
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PATCH_SIZE = 96
DEFAULT_STRIDE = 48
HDF5_CHUNK_SIZE = 128

DEFAULT_INFERENCE_UPSCALE_FACTOR = 2

DATASET_NAME = "DIV2K_train_HR"
DATASET_BASE_DIR = "../Shared"
DATASET_DIR = os.path.join(DATASET_BASE_DIR, DATASET_NAME)
DATASET_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
PREPROCESSED_DATA_DIR = DATASET_BASE_DIR 
HDF5_FILENAME_PATTERN = "div2k_train_patches_x{}.h5"

IMAGE_DISPLAY_SIZE = (300, 300)
CROP_SOURCE_SIZE = (64, 64)
CROP_DISPLAY_SIZE = (128, 128)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_DATALOADER_WORKERS = min(4, os.cpu_count()) if os.cpu_count() and os.name != 'nt' else 0 # 0 for Windows often safer