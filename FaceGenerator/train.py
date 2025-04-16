import os
import threading
import queue
import time
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

DEFAULT_IMAGE_SIZE = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_LATENT_DIM = 100
DEFAULT_EPOCHS = 25
DEFAULT_LR = 0.0002
DEFAULT_BETA1 = 0.5
DEFAULT_NUM_CHANNELS = 3
DEFAULT_NGF = 64
DEFAULT_NDF = 64

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):
    def __init__(self, nz=DEFAULT_LATENT_DIM, ngf=DEFAULT_NGF, nc=DEFAULT_NUM_CHANNELS):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.main = nn.Sequential(
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, input):
        if input.dim() == 2:
             input = input.unsqueeze(-1).unsqueeze(-1)
        elif input.dim() != 4 or input.shape[2] != 1 or input.shape[3] != 1:
             raise ValueError(f"Generator input must be [N, nz, 1, 1], got {input.shape}")
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf=DEFAULT_NDF, nc=DEFAULT_NUM_CHANNELS):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.apply(weights_init)

    def forward(self, input):
        return self.main(input)

class TrainingThread(threading.Thread):
    def __init__(self, app_instance, config, log_queue, status_queue, stop_event):
        super().__init__()
        self.app = app_instance
        self.config = config
        self.log_queue = log_queue
        self.status_queue = status_queue
        self.stop_event = stop_event
        self.daemon = True

    def log(self, message):
        self.log_queue.put(message)

    def update_status(self, message):
        self.status_queue.put(message)

    def run(self):
        moved_files = []
        needs_dummy_folder = False
        dummy_folder = ""
        try:
            self.update_status("Initializing...")
            self.log("Training thread started.")

            dataroot = self.config['dataroot']
            image_size = self.config['image_size']
            batch_size = self.config['batch_size']
            nz = self.config['nz']
            num_epochs = self.config['num_epochs']
            lr = self.config['lr']
            beta1 = self.config['beta1']
            ngf = self.config.get('ngf', DEFAULT_NGF)
            ndf = self.config.get('ndf', DEFAULT_NDF)
            nc = self.config.get('nc', DEFAULT_NUM_CHANNELS)
            load_G_path = self.config.get('load_G_path', None)
            load_D_path = self.config.get('load_D_path', None)
            save_dir = self.config.get('save_dir', '.')

            os.makedirs(save_dir, exist_ok=True)
            model_G_save_path = os.path.join(save_dir, "generator.pth")
            model_D_save_path = os.path.join(save_dir, "discriminator.pth")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.log(f"Using device: {device}")

            self.update_status("Loading dataset...")
            self.log(f"Loading dataset from: {dataroot}")
            if not os.path.isdir(dataroot):
                 self.log(f"ERROR: Dataset directory not found: {dataroot}")
                 self.update_status(f"Error: Dataset not found")
                 return

            subdirs = [d for d in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, d))]
            dataroot_for_imagefolder = dataroot
            if not subdirs:
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) for f in os.listdir(dataroot)):
                    needs_dummy_folder = True
                    dummy_folder = os.path.join(dataroot, "_imgs_temp_")
                    try:
                        os.makedirs(dummy_folder, exist_ok=True)
                        self.log("Images found in root. Moving to temporary '_imgs_temp_' subfolder.")
                        moved_files = []
                        for fname in os.listdir(dataroot):
                           fpath = os.path.join(dataroot, fname)
                           if os.path.isfile(fpath) and fname != "_imgs_temp_" and \
                              fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                              try:
                                   dest_path = os.path.join(dummy_folder, fname)
                                   os.rename(fpath, dest_path)
                                   moved_files.append((dest_path, fpath))
                              except OSError as e:
                                   self.log(f"Warning: Could not move {fname}: {e}")
                        self.log(f"Moved {len(moved_files)} files.")
                        dataroot_for_imagefolder = dummy_folder
                    except Exception as e:
                        self.log(f"Error creating/moving to dummy folder: {e}. Training might fail.")
                        needs_dummy_folder = False
                        dataroot_for_imagefolder = dataroot
            try:
                dataset = dset.ImageFolder(root=dataroot_for_imagefolder,
                                      transform=transforms.Compose([
                                          transforms.Resize(image_size),
                                          transforms.CenterCrop(image_size),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                         shuffle=True, num_workers=2, pin_memory=True)
            except Exception as e:
                 self.log(f"ERROR creating Dataset/DataLoader: {e}")
                 self.update_status(f"Error: Dataset loading failed")
                 if needs_dummy_folder: self._cleanup_dummy_folder(dummy_folder, moved_files)
                 return

            if len(dataset) == 0:
                 self.log("ERROR: No images found in the dataset directory or subdirectories.")
                 self.update_status("Error: Dataset empty")
                 if needs_dummy_folder: self._cleanup_dummy_folder(dummy_folder, moved_files)
                 return

            self.log(f"Dataset loaded. Found {len(dataset)} images.")

            self.update_status("Creating models...")
            netG = Generator(nz, ngf, nc).to(device)
            netD = Discriminator(ndf, nc).to(device)

            if load_G_path and os.path.exists(load_G_path):
                try:
                    netG.load_state_dict(torch.load(load_G_path, map_location=device))
                    self.log(f"Loaded Generator state from: {load_G_path}")
                except Exception as e:
                    self.log(f"Warning: Error loading Generator state from {load_G_path}: {e}. Initializing new weights.")
                    netG.apply(weights_init)
            else:
                 if load_G_path: self.log(f"Generator model not found at {load_G_path}. Initializing new weights.")
                 else: self.log("Initializing new Generator weights.")
                 netG.apply(weights_init)

            if load_D_path and os.path.exists(load_D_path):
                 try:
                    netD.load_state_dict(torch.load(load_D_path, map_location=device))
                    self.log(f"Loaded Discriminator state from: {load_D_path}")
                 except Exception as e:
                    self.log(f"Warning: Error loading Discriminator state from {load_D_path}: {e}. Initializing new weights.")
                    netD.apply(weights_init)
            else:
                 if load_D_path: self.log(f"Discriminator model not found at {load_D_path}. Initializing new weights.")
                 else: self.log("Initializing new Discriminator weights.")
                 netD.apply(weights_init)

            self.log("Models created/loaded.")

            criterion = nn.BCELoss()
            real_label = 1.
            fake_label = 0.
            optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
            optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

            self.log("Starting Training Loop...")
            self.update_status(f"Training Epoch 1/{num_epochs}")
            iters = 0

            for epoch in range(num_epochs):
                epoch_start_time = time.time()
                self.update_status(f"Training Epoch {epoch+1}/{num_epochs}")

                for i, data in enumerate(dataloader, 0):
                    if self.stop_event.is_set():
                        self.log("Stop signal received. Halting training.")
                        self.update_status("Stopped")
                        self.log(f"Saving models at epoch {epoch+1}, iter {i}...")
                        torch.save(netG.state_dict(), model_G_save_path)
                        torch.save(netD.state_dict(), model_D_save_path)
                        self.log(f"Models saved to {save_dir}")
                        if needs_dummy_folder: self._cleanup_dummy_folder(dummy_folder, moved_files)
                        return

                    netD.zero_grad()
                    real_cpu = data[0].to(device)
                    b_size = real_cpu.size(0)
                    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
                    output = netD(real_cpu).view(-1)
                    errD_real = criterion(output, label)
                    errD_real.backward()
                    D_x = output.mean().item()
                    noise = torch.randn(b_size, nz, 1, 1, device=device)
                    fake = netG(noise)
                    label.fill_(fake_label)
                    output = netD(fake.detach()).view(-1)
                    errD_fake = criterion(output, label)
                    errD_fake.backward()
                    D_G_z1 = output.mean().item()
                    errD = errD_real + errD_fake
                    optimizerD.step()

                    netG.zero_grad()
                    label.fill_(real_label)
                    output = netD(fake).view(-1)
                    errG = criterion(output, label)
                    errG.backward()
                    D_G_z2 = output.mean().item()
                    optimizerG.step()

                    if i % 50 == 0:
                        self.log(f'[{epoch+1}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

                    iters += 1

                epoch_time = time.time() - epoch_start_time
                self.log(f"Epoch {epoch+1} finished in {epoch_time:.2f} seconds.")

            self.log("Training finished.")
            self.update_status("Training Complete")

            self.log("Saving final models...")
            torch.save(netG.state_dict(), model_G_save_path)
            torch.save(netD.state_dict(), model_D_save_path)
            self.log(f"Final models saved to {save_dir}")
            if hasattr(self.app, 'update_last_trained_paths'):
                self.app.update_last_trained_paths(model_G_save_path, model_D_save_path)

        except Exception as e:
            self.log(f"FATAL ERROR during training: {e}")
            self.log(traceback.format_exc())
            self.update_status(f"Error: Training failed ({type(e).__name__})")

        finally:
             self.stop_event.clear()
             self.update_status("FINISHED")
             if needs_dummy_folder:
                 self._cleanup_dummy_folder(dummy_folder, moved_files)

    def _cleanup_dummy_folder(self, dummy_folder_path, files_to_move_back):
        if not dummy_folder_path or not files_to_move_back:
            return
        self.log("Cleaning up temporary image folder...")
        try:
            cleaned = False
            for dest, orig in files_to_move_back:
                if os.path.exists(dest):
                    try:
                        os.rename(dest, orig)
                        cleaned = True
                    except OSError as move_err:
                        self.log(f"Warning: Error moving file back during cleanup: {move_err}")
            if cleaned and os.path.exists(dummy_folder_path):
                 try:
                    os.rmdir(dummy_folder_path)
                    self.log("Cleanup complete.")
                 except OSError as rmdir_err:
                     self.log(f"Warning: Could not remove dummy folder {dummy_folder_path}: {rmdir_err}")
        except Exception as clean_e:
            self.log(f"Warning: Error during final cleanup: {clean_e}")