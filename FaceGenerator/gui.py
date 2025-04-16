import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import threading
import queue
import time
from PIL import Image, ImageTk, ImageOps
import numpy as np
import torch
import torchvision.utils as vutils
import traceback

from train import (TrainingThread, Generator, DEFAULT_IMAGE_SIZE, DEFAULT_BATCH_SIZE,
                   DEFAULT_LATENT_DIM, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_BETA1,
                   DEFAULT_NUM_CHANNELS, DEFAULT_NGF, DEFAULT_NDF)

class FaceGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PyTorch Face Generator")
        self.geometry("750x700")

        self.dataset_path = tk.StringVar()
        self.image_size = tk.IntVar(value=DEFAULT_IMAGE_SIZE)
        self.batch_size = tk.IntVar(value=DEFAULT_BATCH_SIZE)
        self.latent_dim = tk.IntVar(value=DEFAULT_LATENT_DIM)
        self.num_epochs = tk.IntVar(value=DEFAULT_EPOCHS)
        self.learning_rate = tk.DoubleVar(value=DEFAULT_LR)
        self.beta1 = tk.DoubleVar(value=DEFAULT_BETA1)
        self.status_var = tk.StringVar(value="Idle")
        self.continue_training_var = tk.BooleanVar(value=False)
        self.generator_model_path = tk.StringVar()

        self.training_thread = None
        self.stop_training_event = threading.Event()
        self.log_queue = queue.Queue()
        self.status_queue = queue.Queue()

        self.last_trained_G_path = "generator.pth"
        self.last_trained_D_path = "discriminator.pth"

        self.generated_photo = None
        self.upscaled_photo = None
        self.last_upscaled_pil_image = None 

        self.create_widgets()
        self.after(100, self.process_queues)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)


    def create_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        train_tab = ttk.Frame(notebook)
        inference_tab = ttk.Frame(notebook)

        notebook.add(train_tab, text='Training')
        notebook.add(inference_tab, text='Inference')

        self._create_training_tab(train_tab)
        self._create_inference_tab(inference_tab)

    def _create_training_tab(self, parent_tab):
        train_frame = ttk.LabelFrame(parent_tab, text="Training Configuration")
        train_frame.pack(pady=10, padx=10, fill=tk.X)
        train_frame.columnconfigure(1, weight=1)

        ttk.Label(train_frame, text="Dataset Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(train_frame, textvariable=self.dataset_path, width=40).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(train_frame, text="Browse...", command=self.browse_dataset).grid(row=0, column=2, padx=5, pady=5)

        hp_frame = ttk.LabelFrame(parent_tab, text="Hyperparameters")
        hp_frame.pack(pady=10, padx=10, fill=tk.X)

        ttk.Label(hp_frame, text="Image Size:").grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(hp_frame, textvariable=self.image_size, width=7).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(hp_frame, text="Batch Size:").grid(row=0, column=2, padx=(20,5), pady=2, sticky="w")
        ttk.Entry(hp_frame, textvariable=self.batch_size, width=7).grid(row=0, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(hp_frame, text="Latent Dim (nz):").grid(row=1, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(hp_frame, textvariable=self.latent_dim, width=7).grid(row=1, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(hp_frame, text="Epochs:").grid(row=1, column=2, padx=(20,5), pady=2, sticky="w")
        ttk.Entry(hp_frame, textvariable=self.num_epochs, width=7).grid(row=1, column=3, padx=5, pady=2, sticky="w")

        ttk.Label(hp_frame, text="Learning Rate:").grid(row=2, column=0, padx=5, pady=2, sticky="w")
        ttk.Entry(hp_frame, textvariable=self.learning_rate, width=10).grid(row=2, column=1, padx=5, pady=2, sticky="w")
        ttk.Label(hp_frame, text="Adam Beta1:").grid(row=2, column=2, padx=(20,5), pady=2, sticky="w")
        ttk.Entry(hp_frame, textvariable=self.beta1, width=7).grid(row=2, column=3, padx=5, pady=2, sticky="w")

        ttk.Checkbutton(hp_frame, text="Continue from last saved models (generator.pth/discriminator.pth)?",
                        variable=self.continue_training_var).grid(row=3, column=0, columnspan=4, padx=5, pady=5, sticky="w")

        controls_frame = ttk.Frame(parent_tab)
        controls_frame.pack(pady=5, padx=10, fill=tk.X)

        self.start_button = ttk.Button(controls_frame, text="Start Training", command=self.start_training)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(controls_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Label(controls_frame, text="Status:").pack(side=tk.LEFT, padx=(10, 2))
        self.status_label = ttk.Label(controls_frame, textvariable=self.status_var, relief=tk.SUNKEN, width=30)
        self.status_label.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        log_frame = ttk.LabelFrame(parent_tab, text="Logs")
        log_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.log_area = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10, state=tk.DISABLED)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _create_inference_tab(self, parent_tab):
        inference_frame = ttk.LabelFrame(parent_tab, text="Generate & Upscale Face")
        inference_frame.pack(pady=10, padx=10, fill=tk.X)
        inference_frame.columnconfigure(1, weight=1)

        ttk.Label(inference_frame, text="Generator Model:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(inference_frame, textvariable=self.generator_model_path, width=40).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(inference_frame, text="Browse...", command=self.browse_generator_model).grid(row=0, column=2, padx=5, pady=5)

        # Frame for buttons
        inf_buttons_frame = ttk.Frame(inference_frame)
        inf_buttons_frame.grid(row=1, column=0, columnspan=3, pady=10)

        self.generate_button = ttk.Button(inf_buttons_frame, text="Generate & Upscale Face (Bicubic)", command=self.generate_and_upscale_face)
        self.generate_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(inf_buttons_frame, text="Save Upscaled Image", command=self.save_upscaled_image, state=tk.DISABLED) # Initially disabled
        self.save_button.pack(side=tk.LEFT, padx=5)

        display_frame = ttk.Frame(parent_tab)
        display_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        gen_img_frame = ttk.LabelFrame(display_frame, text="Generated (Low Res)")
        gen_img_frame.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        low_res_size = self.image_size.get() if self.image_size.get() > 0 else DEFAULT_IMAGE_SIZE
        self.image_label_gen = ttk.Label(gen_img_frame, text=f"Output ({low_res_size}x{low_res_size})")
        self.image_label_gen.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

        upscale_img_frame = ttk.LabelFrame(display_frame, text="Upscaled (4x Bicubic)")
        upscale_img_frame.pack(side=tk.RIGHT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        high_res_size = low_res_size * 4
        self.image_label_upscaled = ttk.Label(upscale_img_frame, text=f"Output ({high_res_size}x{high_res_size})")
        self.image_label_upscaled.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

    def browse_dataset(self):
        path = filedialog.askdirectory(title="Select Dataset Folder (containing images or image subfolders)")
        if path:
            self.dataset_path.set(path)
            self.log_message(f"Dataset path set to: {path}")

    def browse_generator_model(self):
        path = filedialog.askopenfilename(
            title="Select Generator Model File",
            filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")]
        )
        if path:
            self.generator_model_path.set(path)
            self.log_message(f"Generator model for inference set to: {path}")

    def update_last_trained_paths(self, g_path, d_path):
        self.last_trained_G_path = g_path
        self.last_trained_D_path = d_path
        if not self.generator_model_path.get():
             self.generator_model_path.set(g_path)
             self.log_message(f"Default inference model updated to: {g_path}")

    def start_training(self):
        if self.training_thread and self.training_thread.is_alive():
            messagebox.showwarning("Training Active", "Training is already in progress.")
            return

        dataroot = self.dataset_path.get()
        if not dataroot:
            messagebox.showerror("Error", "Please select a dataset folder first.")
            return
        if not os.path.isdir(dataroot):
             messagebox.showerror("Error", f"Dataset folder not found:\n{dataroot}")
             return

        self.save_button.config(state=tk.DISABLED)
        self.last_upscaled_pil_image = None

        try:
            config = {
                'dataroot': dataroot,
                'image_size': self.image_size.get(),
                'batch_size': self.batch_size.get(),
                'nz': self.latent_dim.get(),
                'num_epochs': self.num_epochs.get(),
                'lr': self.learning_rate.get(),
                'beta1': self.beta1.get(),
                'nc': DEFAULT_NUM_CHANNELS,
                'ngf': DEFAULT_NGF,
                'ndf': DEFAULT_NDF,
                'save_dir': '.'
            }
            if not all(v > 0 for k, v in config.items() if k in ['image_size', 'batch_size', 'nz', 'num_epochs', 'lr']):
                 raise ValueError("Numeric hyperparameters must be positive.")
            if not (0 < config['beta1'] < 1):
                 raise ValueError("Adam Beta1 must be between 0 and 1.")

        except (tk.TclError, ValueError) as e:
             messagebox.showerror("Invalid Input", f"Please check hyperparameter values.\nError: {e}")
             return

        if self.continue_training_var.get():
            g_path_to_check = self.last_trained_G_path
            d_path_to_check = self.last_trained_D_path
            if os.path.exists(g_path_to_check) and os.path.exists(d_path_to_check):
                 config['load_G_path'] = g_path_to_check
                 config['load_D_path'] = d_path_to_check
                 self.log_message(f"Attempting to continue training from '{g_path_to_check}' and '{d_path_to_check}'.")
            else:
                 messagebox.showwarning("Continue Failed", f"Could not find '{g_path_to_check}' or '{d_path_to_check}'.\nStarting new training session.")
                 self.continue_training_var.set(False)

        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete('1.0', tk.END)
        self.log_area.config(state=tk.DISABLED)
        self.log_message("--- Starting Training ---")
        self.log_message(f"Config: {config}")

        self.stop_training_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Starting...")

        self.training_thread = TrainingThread(
            self, config, self.log_queue, self.status_queue, self.stop_training_event
        )
        self.training_thread.start()

    def stop_training(self):
        if self.training_thread and self.training_thread.is_alive():
            self.log_message("Stop requested. Signalling training thread...")
            self.stop_training_event.set()
            self.stop_button.config(state=tk.DISABLED)
            self.status_var.set("Stopping...")
        else:
            self.log_message("No active training process to stop.")
            self.stop_button.config(state=tk.DISABLED)
            self.start_button.config(state=tk.NORMAL)


    def generate_and_upscale_face(self):
        model_path = self.generator_model_path.get()
        if not model_path:
            if os.path.exists(self.last_trained_G_path):
                model_path = self.last_trained_G_path
                self.generator_model_path.set(model_path)
                self.log_message(f"Using last trained generator: {model_path}")
            else:
                messagebox.showerror("Error", "Please select a Generator model (.pth file) or train one first.")
                self.save_button.config(state=tk.DISABLED) # Ensure save is disabled
                self.last_upscaled_pil_image = None
                return
        elif not os.path.exists(model_path):
             messagebox.showerror("Error", f"Generator model file not found:\n{model_path}")
             self.save_button.config(state=tk.DISABLED) # Ensure save is disabled
             self.last_upscaled_pil_image = None
             return

        self.save_button.config(state=tk.DISABLED)
        self.last_upscaled_pil_image = None

        try:
            self.status_var.set("Loading generator...")
            self.update_idletasks()

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            try:
                nz = self.latent_dim.get()
                img_size_gen = self.image_size.get()
                nc = DEFAULT_NUM_CHANNELS
                ngf = DEFAULT_NGF
                if img_size_gen <= 0 or nz <= 0: raise ValueError("Invalid dimensions")
            except (tk.TclError, ValueError) as e:
                 messagebox.showerror("Inference Error", f"Invalid Latent Dim or Image Size set in GUI for model loading.\nError: {e}")
                 self.status_var.set("Error")
                 return

            self.log_message(f"Loading Generator (nz={nz}, ngf={ngf}, nc={nc}) for {img_size_gen}x{img_size_gen} output.")
            netG = Generator(nz, ngf, nc).to(device)
            netG.load_state_dict(torch.load(model_path, map_location=device))
            netG.eval()

            self.status_var.set("Generating low-res...")
            self.update_idletasks()

            noise = torch.randn(1, nz, 1, 1, device=device)
            with torch.no_grad():
                fake_image_tensor = netG(noise).detach().cpu()

            img_gen_np = vutils.make_grid(fake_image_tensor, padding=0, normalize=True).numpy()
            img_gen_pil = Image.fromarray(np.transpose(img_gen_np * 255, (1, 2, 0)).astype(np.uint8))

            display_size_low = 128
            img_gen_display = img_gen_pil.copy()
            img_gen_display.thumbnail((display_size_low, display_size_low), Image.Resampling.LANCZOS)
            self.generated_photo = ImageTk.PhotoImage(img_gen_display)
            self.image_label_gen.config(image=self.generated_photo, text="")
            self.image_label_gen.image = self.generated_photo
            self.image_label_gen.master.config(text=f"Generated ({img_size_gen}x{img_size_gen})")

            self.status_var.set("Upscaling (Bicubic)...")
            self.log_message("Starting 4x Bicubic upscale...")
            self.update_idletasks()

            target_size = (img_size_gen * 4, img_size_gen * 4)
            img_upscaled_pil = img_gen_pil.resize(target_size, Image.Resampling.BICUBIC)
            self.log_message("Bicubic upscaling complete.")

            self.last_upscaled_pil_image = img_upscaled_pil

            display_size_high = 300
            img_upscaled_display = img_upscaled_pil.copy()
            img_upscaled_display.thumbnail((display_size_high, display_size_high), Image.Resampling.LANCZOS)
            self.upscaled_photo = ImageTk.PhotoImage(img_upscaled_display)
            self.image_label_upscaled.config(image=self.upscaled_photo, text="")
            self.image_label_upscaled.image = self.upscaled_photo
            self.image_label_upscaled.master.config(text=f"Upscaled ({target_size[0]}x{target_size[1]})")

            self.status_var.set("Idle")
            self.log_message("Generation and Bicubic upscaling successful.")
            self.save_button.config(state=tk.NORMAL)

        except FileNotFoundError:
            messagebox.showerror("Error", f"Generator model file not found: {model_path}")
            self.status_var.set("Error")
            self.log_message(f"Error: Generator model not found at {model_path}")
        except RuntimeError as e:
             messagebox.showerror("Inference Error", f"Model loading failed. Architecture mismatch?\n(Check Latent Dim, Image Size, Features in GUI vs. trained model)\nDetails: {e}")
             self.log_message(f"ERROR loading generator state: {e}")
             self.status_var.set("Error")
        except Exception as e:
            messagebox.showerror("Inference Error", f"An unexpected error occurred:\n{e}")
            self.log_message(f"ERROR during generation/upscaling: {e}")
            self.log_message(traceback.format_exc())
            self.status_var.set("Error")

    def save_upscaled_image(self):
        if not self.last_upscaled_pil_image:
            messagebox.showwarning("Save Error", "No upscaled image available to save. Please generate one first.")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Upscaled Image As",
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )

        if not file_path:
            self.log_message("Save cancelled by user.")
            return

        try:
            self.log_message(f"Saving upscaled image to: {file_path}")
            self.last_upscaled_pil_image.save(file_path)
            self.log_message("Image saved successfully.")
            messagebox.showinfo("Save Successful", f"Image saved to:\n{file_path}")
        except Exception as e:
            self.log_message(f"ERROR saving image: {e}")
            self.log_message(traceback.format_exc())
            messagebox.showerror("Save Error", f"Failed to save image to:\n{file_path}\n\nError: {e}")


    def process_queues(self):
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_message(msg)
        except queue.Empty:
            pass

        try:
            while True:
                 status_msg = self.status_queue.get_nowait()
                 if status_msg.startswith("Error: Dataset not found"):
                      messagebox.showerror("Training Error", f"Dataset directory not found.\nPlease check the path.")
                      self.status_var.set("Error: Check Dataset Path")
                      self.save_button.config(state=tk.DISABLED) # Disable save on error
                      self.last_upscaled_pil_image = None
                 elif status_msg.startswith("Error: Dataset empty"):
                      messagebox.showerror("Training Error", f"No images found in the specified dataset directory.")
                      self.status_var.set("Error: Dataset Empty")
                      self.save_button.config(state=tk.DISABLED)
                      self.last_upscaled_pil_image = None
                 elif status_msg.startswith("Error: Dataset loading failed"):
                      messagebox.showerror("Training Error", f"Failed to load dataset. Check logs for details.")
                      self.status_var.set("Error: Dataset Load Fail")
                      self.save_button.config(state=tk.DISABLED)
                      self.last_upscaled_pil_image = None
                 elif status_msg.startswith("Error: Training failed"):
                      messagebox.showerror("Training Error", f"Training failed unexpectedly. Check logs for details.\n({status_msg})")
                      self.status_var.set(status_msg)
                      self.save_button.config(state=tk.DISABLED)
                      self.last_upscaled_pil_image = None
                 elif status_msg == "FINISHED":
                     is_stopped_by_user = self.stop_training_event.is_set()
                     current_status = self.status_var.get()

                     self.start_button.config(state=tk.NORMAL)
                     self.stop_button.config(state=tk.DISABLED)
                     self.training_thread = None
                     self.stop_training_event.clear()

                     if is_stopped_by_user:
                         self.status_var.set("Stopped")
                     elif "Error" not in current_status:
                         self.status_var.set("Training Complete")

                     self.log_message("--- Training thread finished ---")

                 else:
                     self.status_var.set(status_msg)

        except queue.Empty:
            pass
        self.after(150, self.process_queues)

    def log_message(self, message):
        try:
            if self.log_area.winfo_exists():
                self.log_area.config(state=tk.NORMAL)
                self.log_area.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
                self.log_area.see(tk.END)
                self.log_area.config(state=tk.DISABLED)
        except tk.TclError:
            pass

    def on_closing(self):
        if self.training_thread and self.training_thread.is_alive():
            if messagebox.askokcancel("Quit", "Training is in progress. Stop training and quit?"):
                self.log_message("Quit requested during training. Stopping thread...")
                self.stop_training()
                self.after(1000, self.destroy)
            else:
                return
        else:
            self.destroy()