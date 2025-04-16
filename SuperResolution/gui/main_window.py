import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import threading
import os
import time
import queue 

import config
from models.srcnn import SRCNN
from utils.file_utils import parse_scale_from_filename
from data.download import check_and_download_dataset
from data.preparation import prepare_training_data_incremental
from core.upscaler import upscale_image, upscale_bicubic
from core.trainer import train_model
from gui.utils import display_image_in_label 

class SuperResolutionApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("SRCNN Image Upscaler")
        self.geometry("950x880") 
        self.device = config.DEVICE
        self.model = None
        self.loaded_model_scale = None
        self.input_image_path = None
        self.input_image_pil = None
        self.output_image_pil = None
        self.bicubic_comparison_pil = None
        self.original_display_thumb = None
        self.output_display_thumb = None  
        self.training_in_progress = False
        self.upscaling_in_progress = False
        self.dataset_available = False
        self.current_train_data_hdf5_path = None
        self.status_update_queue = queue.Queue() 

        self.model_path = tk.StringVar(value=config.DEFAULT_MODEL_FILENAME_PATTERN.format(config.DEFAULT_TRAIN_UPSCALE_FACTOR))
        self.epochs_var = tk.IntVar(value=config.DEFAULT_EPOCHS)
        self.batch_size_var = tk.IntVar(value=config.DEFAULT_BATCH_SIZE)
        self.batch_limit_var = tk.IntVar(value=config.DEFAULT_BATCH_LIMIT)
        self.lr_var = tk.DoubleVar(value=config.DEFAULT_LEARNING_RATE)
        self.train_upscale_factor_var = tk.IntVar(value=config.DEFAULT_TRAIN_UPSCALE_FACTOR)
        self.inference_upscale_factor_var = tk.IntVar(value=config.DEFAULT_INFERENCE_UPSCALE_FACTOR)
        self.status_var = tk.StringVar(value="Initializing...")

        self.create_widgets()

        self.check_status_queue() 
        self.after(100, self.initial_setup) 

    def update_status(self, message, error=False, warning=False):
        self.status_update_queue.put((message, error, warning))

    def check_status_queue(self):
        try:
            while True: 
                message, error, warning = self.status_update_queue.get_nowait()
                self._update_status_display(message, error=error, warning=warning)
        except queue.Empty:
            pass
        finally:
            self.after(100, self.check_status_queue)

    def _update_status_display(self, message, error=False, warning=False):
        print(f"STATUS: {'ERROR: ' if error else 'WARN: ' if warning else ''}{message}") # Also print
        self.status_var.set(message)
        fg_color = "black"
        if error: fg_color = "red"
        elif warning: fg_color = "orange"
        if hasattr(self, 'status_bar'):
            self.status_bar.config(foreground=fg_color)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.rowconfigure(2, weight=1) 
        main_frame.rowconfigure(3, weight=0) 
        main_frame.rowconfigure(4, weight=0) 
        main_frame.columnconfigure(0, weight=1)

        top_frame = ttk.LabelFrame(main_frame, text="Model & Training Settings", padding="10")
        top_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        top_frame.columnconfigure(1, weight=1)

        ttk.Label(top_frame, text="Model Path:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        self.model_entry = ttk.Entry(top_frame, textvariable=self.model_path, width=50)
        self.model_entry.grid(row=0, column=1, sticky="ew", padx=5)
        self.browse_model_button = ttk.Button(top_frame, text="Browse...", command=self.browse_model)
        self.browse_model_button.grid(row=0, column=2, padx=(0, 5))
        self.load_model_button = ttk.Button(top_frame, text="Load Model", command=self.load_model)
        self.load_model_button.grid(row=0, column=3, padx=5)

        param_frame = ttk.Frame(top_frame)
        param_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(5,0))

        ttk.Label(param_frame, text="Train Scale:").pack(side=tk.LEFT, padx=(0, 2))
        self.train_scale_spinbox = ttk.Spinbox(param_frame, from_=2, to=8, increment=1, width=3,
                                               textvariable=self.train_upscale_factor_var, state=tk.DISABLED,
                                               command=self._update_train_button_text) 
        self.train_scale_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(param_frame, text="Epochs:").pack(side=tk.LEFT, padx=(0, 2))
        self.epochs_spinbox = ttk.Spinbox(param_frame, from_=1, to=1000, increment=1, width=5,
                                          textvariable=self.epochs_var, state=tk.DISABLED)
        self.epochs_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(param_frame, text="Batch Size:").pack(side=tk.LEFT, padx=(0, 2))
        self.batch_size_spinbox = ttk.Spinbox(param_frame, from_=1, to=1024, increment=1, width=5,
                                             textvariable=self.batch_size_var, state=tk.DISABLED)
        self.batch_size_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(param_frame, text="Batch Limit (0=None):").pack(side=tk.LEFT, padx=(0, 2))
        self.batch_limit_spinbox = ttk.Spinbox(param_frame, from_=0, to=100000, increment=1, width=7, 
                                              textvariable=self.batch_limit_var, state=tk.DISABLED)
        self.batch_limit_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(param_frame, text="Learn Rate:").pack(side=tk.LEFT, padx=(0, 2))
        self.lr_spinbox = ttk.Spinbox(param_frame, from_=1e-7, to=1e-1, increment=1e-5, width=8, 
                                      textvariable=self.lr_var, format="%.1e", state=tk.DISABLED)
        self.lr_spinbox.pack(side=tk.LEFT, padx=(0, 15))

        self.train_button = ttk.Button(param_frame, text=f"Train (for x{config.DEFAULT_TRAIN_UPSCALE_FACTOR})",
                                       command=self.start_training_sequence, state=tk.DISABLED)
        self.train_button.pack(side=tk.LEFT, padx=5)

        mid_frame = ttk.LabelFrame(main_frame, text="Image Operations", padding="10")
        mid_frame.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.load_image_button = ttk.Button(mid_frame, text="Load Input Image", command=self.load_image)
        self.load_image_button.pack(side=tk.LEFT, padx=5)

        ttk.Label(mid_frame, text="Upscale Factor:").pack(side=tk.LEFT, padx=(10, 2))
        self.upscale_factor_spinbox = ttk.Spinbox(mid_frame, from_=2, to=8, increment=1, width=3,
                                                  textvariable=self.inference_upscale_factor_var, state=tk.DISABLED,
                                                  command=self._update_upscale_button_state) 
        self.upscale_factor_spinbox.pack(side=tk.LEFT, padx=(0, 10))

        self.upscale_button = ttk.Button(mid_frame, text="Upscale Image", command=self.start_upscale_sequence, state=tk.DISABLED)
        self.upscale_button.pack(side=tk.LEFT, padx=5)
        self.save_button = ttk.Button(mid_frame, text="Save Upscaled Image", command=self.save_image, state=tk.DISABLED)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.save_bicubic_button = ttk.Button(mid_frame, text="Save Comparison (Bicubic)", command=self.save_bicubic_image, state=tk.DISABLED)
        self.save_bicubic_button.pack(side=tk.LEFT, padx=5)

        display_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        display_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 5))
        display_frame.columnconfigure(0, weight=1); display_frame.columnconfigure(1, weight=1)
        display_frame.rowconfigure(1, weight=1)

        self.original_title_label = ttk.Label(display_frame, text="Comparison Image", anchor=tk.CENTER)
        self.original_title_label.grid(row=0, column=0, pady=(0, 5), sticky="ew")
        self.upscaled_title_label = ttk.Label(display_frame, text="Output Image", anchor=tk.CENTER)
        self.upscaled_title_label.grid(row=0, column=1, pady=(0, 5), sticky="ew")

        orig_img_frame = ttk.Frame(display_frame, borderwidth=1, relief=tk.SUNKEN)
        orig_img_frame.grid(row=1, column=0, sticky="nsew", padx=5)
        orig_img_frame.rowconfigure(0, weight=1); orig_img_frame.columnconfigure(0, weight=1) 
        self.original_image_label = ttk.Label(orig_img_frame, background="lightgrey", anchor=tk.CENTER, text="Load an image")
        self.original_image_label.grid(row=0, column=0, sticky="nsew")
        self.original_image_label.bind("<Motion>", self._handle_mouse_motion)
        self.original_image_label.bind("<Leave>", self._handle_mouse_leave)

        upscaled_img_frame = ttk.Frame(display_frame, borderwidth=1, relief=tk.SUNKEN)
        upscaled_img_frame.grid(row=1, column=1, sticky="nsew", padx=5)
        upscaled_img_frame.rowconfigure(0, weight=1); upscaled_img_frame.columnconfigure(0, weight=1)
        self.upscaled_image_label = ttk.Label(upscaled_img_frame, background="lightgrey", anchor=tk.CENTER, text="Output appears here")
        self.upscaled_image_label.grid(row=0, column=0, sticky="nsew")

        crop_frame = ttk.Frame(main_frame, padding=(0, 5, 0, 0))
        crop_frame.grid(row=3, column=0, sticky="ew", pady=(5, 5))
        crop_frame.columnconfigure(0, weight=1)
        crop_frame.columnconfigure(1, weight=1)

        ttk.Label(crop_frame, text="Comparison Crop", anchor=tk.CENTER).grid(row=0, column=0)
        ttk.Label(crop_frame, text="Output Crop", anchor=tk.CENTER).grid(row=0, column=1)

        orig_crop_border = ttk.Frame(crop_frame, borderwidth=1, relief=tk.SUNKEN,
                                    width=config.CROP_DISPLAY_SIZE[0], height=config.CROP_DISPLAY_SIZE[1])
        orig_crop_border.grid(row=1, column=0, padx=5, pady=(0,5), sticky="n") 
        orig_crop_border.pack_propagate(False) 
        self.original_crop_label = ttk.Label(orig_crop_border, background="grey", text="N/A", anchor=tk.CENTER)
        self.original_crop_label.pack(fill=tk.BOTH, expand=True)

        upscaled_crop_border = ttk.Frame(crop_frame, borderwidth=1, relief=tk.SUNKEN,
                                        width=config.CROP_DISPLAY_SIZE[0], height=config.CROP_DISPLAY_SIZE[1])
        upscaled_crop_border.grid(row=1, column=1, padx=5, pady=(0,5), sticky="n") 
        upscaled_crop_border.pack_propagate(False) 
        self.upscaled_crop_label = ttk.Label(upscaled_crop_border, background="grey", text="N/A", anchor=tk.CENTER)
        self.upscaled_crop_label.pack(fill=tk.BOTH, expand=True)

        self.empty_crop_image = ImageTk.PhotoImage(Image.new('RGBA', config.CROP_DISPLAY_SIZE, (0, 0, 0, 0)))
        self._clear_crop_views()

        self.status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=(5,2))
        self.status_bar.grid(row=4, column=0, sticky="ew")


    def initial_setup(self):
        self.update_status(f"Using device: {self.device}")
        if str(self.device) == "cuda":
            try:
                 gpu_name = torch.cuda.get_device_name(0)
                 self.update_status(f"GPU detected: {gpu_name}")
            except Exception as e:
                 self.update_status(f"Could not get GPU name: {e}", warning=True)
        else:
            self.update_status("CUDA GPU not detected, using CPU. Training/Upscaling will be slower.", warning=True)

        self.update_status("Checking for DIV2K dataset...")
        thread = threading.Thread(target=self._run_dataset_check_worker, daemon=True)
        thread.start()

    def _run_dataset_check_worker(self):
        is_available = check_and_download_dataset(
            config.DATASET_URL,
            config.DATASET_DIR,
            config.DATASET_NAME,
            self.update_status 
        )
        self.after(0, self._finish_dataset_check, is_available)

    def _finish_dataset_check(self, is_available):
        self.dataset_available = is_available
        if self.dataset_available:
            self.update_status("DIV2K dataset is available.")
            self._set_training_controls_state(tk.NORMAL)
        else:
            self.update_status("DIV2K dataset check failed or cancelled. Training disabled.", warning=True)
            self._set_training_controls_state(tk.DISABLED)

        self.load_model()


    def browse_model(self):
        path = filedialog.askopenfilename(
            title="Select Model File", filetypes=[("PyTorch Weights", "*.pth"), ("All Files", "*.*")])
        if path:
            self.model_path.set(path)
            self.load_model()

    def load_model(self):
        path = self.model_path.get()
        self.model = None 
        self.loaded_model_scale = None

        default_scale = config.DEFAULT_TRAIN_UPSCALE_FACTOR
        default_path_expected = os.path.join(".", config.DEFAULT_MODEL_FILENAME_PATTERN.format(default_scale))

        if not os.path.exists(path):
             is_default_missing = False
             try:
                 norm_path = os.path.normpath(path)
                 norm_default = os.path.normpath(default_path_expected)
                 pattern_match = False
                 for scale in range(2, 9):
                      pattern = os.path.normpath(os.path.join(".", config.DEFAULT_MODEL_FILENAME_PATTERN.format(scale)))
                      if norm_path == pattern:
                           pattern_match = True
                           break
                 if pattern_match:
                     is_default_missing = True
             except Exception:
                 pass

             if is_default_missing:
                  self.update_status(f"Default model pattern '{os.path.basename(path)}' not found. Train or load a model.", warning=True)
             else:
                  self.update_status(f"Model file not found: {path}", warning=True)

             self._update_upscale_button_state() 
             return

        try:
            parsed_scale = parse_scale_from_filename(path)

            self.model = SRCNN(num_channels=1).to(self.device)
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()

            if parsed_scale:
                 self.loaded_model_scale = parsed_scale
                 self.update_status(f"SRCNN model loaded: {os.path.basename(path)} (Detected as x{self.loaded_model_scale}) on {self.device}")
                 self.inference_upscale_factor_var.set(parsed_scale)
            else:
                 self.loaded_model_scale = None 
                 self.update_status(f"SRCNN model loaded: {os.path.basename(path)} (Scale not detected). Set Inference Scale manually.", warning=True)

            self._update_upscale_button_state()

        except FileNotFoundError:
             self.model = None
             self.loaded_model_scale = None
             self.update_status(f"Model file disappeared: {path}", error=True)
             messagebox.showerror("Model Load Error", f"Model file not found at {path}.")
             self._update_upscale_button_state()
        except Exception as e:
            self.model = None
            self.loaded_model_scale = None
            self.update_status(f"Error loading model '{os.path.basename(path)}': {e}", error=True)
            messagebox.showerror("Model Load Error", f"Could not load model weights from {path}.\nError: {e}\n\nCheck model architecture and file integrity.")
            self._update_upscale_button_state()


    def load_image(self):
        path = filedialog.askopenfilename(
            title="Select Input Image", filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff"), ("All Files", "*.*")])
        if not path: return 

        try:
            img = Image.open(path).convert('RGB')
            self.input_image_path = path
            self.input_image_pil = img

            self.output_image_pil = None
            self.bicubic_comparison_pil = None
            self._clear_crop_views()

            self.original_display_thumb = display_image_in_label(self.input_image_pil, self.original_image_label, config.IMAGE_DISPLAY_SIZE)
            self.original_title_label.config(text="Original Image")

            self.output_display_thumb = display_image_in_label(None, self.upscaled_image_label, config.IMAGE_DISPLAY_SIZE)
            self.upscaled_image_label.config(text="Output appears here")
            self.upscaled_title_label.config(text="Output Image")

            self.update_status(f"Loaded image: {os.path.basename(path)} ({self.input_image_pil.width}x{self.input_image_pil.height})")

            self.upscale_factor_spinbox.config(state=tk.NORMAL)
            self.save_button.config(state=tk.DISABLED) 
            self.save_bicubic_button.config(state=tk.DISABLED)
            self._update_upscale_button_state() 

        except FileNotFoundError:
            self.update_status(f"Image file not found: {path}", error=True)
            messagebox.showerror("Image Load Error", f"File not found:\n{path}")
            self._reset_image_state()
        except Exception as e:
            self.update_status(f"Error loading image: {e}", error=True)
            messagebox.showerror("Image Load Error", f"Could not load image file.\nError: {e}")
            self._reset_image_state()

    def _reset_image_state(self):
        self.input_image_path = None
        self.input_image_pil = None
        self.output_image_pil = None
        self.bicubic_comparison_pil = None
        self.original_display_thumb = display_image_in_label(None, self.original_image_label, config.IMAGE_DISPLAY_SIZE)
        self.output_display_thumb = display_image_in_label(None, self.upscaled_image_label, config.IMAGE_DISPLAY_SIZE)
        self.original_image_label.config(text="Load an image")
        self.upscaled_image_label.config(text="Output appears here")
        self.original_title_label.config(text="Comparison Image")
        self.upscaled_title_label.config(text="Output Image")
        self._clear_crop_views()
        self.upscale_factor_spinbox.config(state=tk.DISABLED)
        self.upscale_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.save_bicubic_button.config(state=tk.DISABLED)
            
    def save_image(self):
        if not self.output_image_pil:
            messagebox.showwarning("Save Error", "No upscaled image is available to save.")
            return
        if not self.input_image_path:
             messagebox.showerror("Save Error", "Cannot determine original filename.")
             return # Should not happen if output_image_pil exists, but safety check

        try:
            factor = self.inference_upscale_factor_var.get()
        except tk.TclError:
            factor = config.DEFAULT_INFERENCE_UPSCALE_FACTOR
            self.update_status(f"Could not read inference factor, using default {factor}", warning=True)

        # Determine if SRCNN was likely used for the current output
        use_srcnn = self.model is not None and self.loaded_model_scale == factor
        mode_suffix = "_srcnn" if use_srcnn else "_bicubic"

        base, ext = os.path.splitext(os.path.basename(self.input_image_path))
        # Ensure extension is valid for saving, default to png
        valid_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        if ext.lower() not in valid_exts:
            ext = ".png"

        suggested_name = f"{base}_x{factor}{mode_suffix}{ext}"
        initial_dir = os.path.dirname(self.input_image_path)

        save_path = filedialog.asksaveasfilename(
            title="Save Upscaled Image",
            initialdir=initial_dir,
            initialfile=suggested_name,
            defaultextension=ext, # Use determined extension
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff"), ("All Files", "*.*")])

        if save_path:
            try:
                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.output_image_pil.save(save_path)
                self.update_status(f"Upscaled image saved to {os.path.basename(save_path)}")
            except Exception as e:
                self.update_status(f"Error saving image to {save_path}: {e}", error=True)
                messagebox.showerror("Save Error", f"Could not save image to {save_path}.\nError: {e}")

    def save_bicubic_image(self):
        """Saves the bicubic upscaled comparison image currently displayed."""
        if not self.bicubic_comparison_pil:
            messagebox.showwarning("Save Error", "No bicubic comparison image is available to save.")
            return
        if not self.input_image_path:
             messagebox.showerror("Save Error", "Cannot determine original filename.")
             return

        try:
            # Use the same factor that was used for the current comparison image
            factor = self.inference_upscale_factor_var.get()
        except tk.TclError:
            # Fallback if factor can't be read (shouldn't happen if comparison exists)
            factor = config.DEFAULT_INFERENCE_UPSCALE_FACTOR
            self.update_status(f"Could not read inference factor, using default {factor} for filename", warning=True)

        # Construct suggested filename specifically for bicubic
        mode_suffix = "_bicubic" # Always bicubic for this button

        base, ext = os.path.splitext(os.path.basename(self.input_image_path))
        # Ensure extension is valid for saving, default to png
        valid_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        if ext.lower() not in valid_exts:
            ext = ".png"

        suggested_name = f"{base}_x{factor}{mode_suffix}{ext}"
        initial_dir = os.path.dirname(self.input_image_path)

        save_path = filedialog.asksaveasfilename(
            title="Save Bicubic Comparison Image", # Updated title
            initialdir=initial_dir,
            initialfile=suggested_name,
            defaultextension=ext,
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tiff"), ("All Files", "*.*")])

        if save_path:
            try:
                # Ensure directory exists before saving
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # --- SAVE THE BICUBIC IMAGE ---
                self.bicubic_comparison_pil.save(save_path)
                # --- END SAVE ---
                self.update_status(f"Bicubic comparison image saved to {os.path.basename(save_path)}")
            except Exception as e:
                self.update_status(f"Error saving bicubic image to {save_path}: {e}", error=True)
                messagebox.showerror("Save Error", f"Could not save bicubic image to {save_path}.\nError: {e}")

    def start_upscale_sequence(self):
        """ Starts the upscaling process in a new thread """
        if not self.input_image_pil:
            messagebox.showwarning("Upscale Error", "Please load an input image first.")
            return
        if self.upscaling_in_progress:
            messagebox.showwarning("Busy", "Upscaling is already in progress.")
            return

        try:
            selected_inference_factor = self.inference_upscale_factor_var.get()
            if selected_inference_factor < 2:
                raise ValueError("Upscale factor must be >= 2.")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Invalid Parameter", f"Invalid inference upscale factor: {e}")
            return

        use_srcnn = (self.model is not None and
                     self.loaded_model_scale is not None and
                     selected_inference_factor == self.loaded_model_scale)
        mode = "SRCNN" if use_srcnn else "Bicubic"
        self.update_status(f"Starting upscale x{selected_inference_factor} ({mode})...")

        self.save_bicubic_button.config(state=tk.DISABLED)
        self.upscaling_in_progress = True
        self._set_all_controls_state(tk.DISABLED)
        self._clear_crop_views() 

        thread = threading.Thread(target=self._run_upscale_worker,
                                  args=(self.input_image_pil.copy(), 
                                        selected_inference_factor,
                                        use_srcnn),
                                  daemon=True)
        thread.start()

    def _run_upscale_worker(self, input_img_copy, factor, use_srcnn):
        """ Worker thread for upscaling """
        start_time = time.time()
        final_img = None
        bicubic_comp = None
        mode_used = "Bicubic" # Default
        error = None

        try:
            # Always generate bicubic for comparison
            bicubic_comp = upscale_bicubic(input_img_copy, factor)

            if use_srcnn and self.model:
                mode_used = "SRCNN"
                # Pass the *original* image copy to the SRCNN upscaler
                final_img = upscale_image(input_img_copy, self.model, factor, self.device)
            else:
                # If not using SRCNN, the final image is just the bicubic one
                final_img = bicubic_comp
                mode_used = "Bicubic"
                # Add reason for using Bicubic
                reason = ""
                if self.model is None: reason = "(No model loaded)"
                elif self.loaded_model_scale is None: reason = "(Loaded model scale unknown)"
                elif factor != self.loaded_model_scale: reason = f"(Model scale x{self.loaded_model_scale} != Target x{factor})"
                self.update_status(f"Using Bicubic upscale {reason}.")

        except Exception as e:
            error = e
            print(f"Error during upscale worker: {e}")
            import traceback
            traceback.print_exc()

        elapsed_time = time.time() - start_time

        # Send results back to main thread
        self.after(0, self._finish_upscale, final_img, bicubic_comp, elapsed_time, mode_used, factor, error)

    def _finish_upscale(self, final_img, bicubic_comp, elapsed_time, mode_used, factor, error):
        self.upscaling_in_progress = False

        if error:
            self.update_status(f"Upscaling x{factor} failed: {error}", error=True)
            messagebox.showerror("Upscale Error", f"An error occurred during upscaling x{factor}:\n{error}")
            self.output_image_pil = None
            self.bicubic_comparison_pil = None
            self.output_display_thumb = display_image_in_label(None, self.upscaled_image_label, config.IMAGE_DISPLAY_SIZE)
            self.original_display_thumb = display_image_in_label(self.input_image_pil, self.original_image_label, config.IMAGE_DISPLAY_SIZE) # Show original again
            self.upscaled_image_label.config(text="Upscaling Failed")
            self.original_title_label.config(text="Original Image")
            self.upscaled_title_label.config(text="Output Image")
            self.save_button.config(state=tk.DISABLED)
            self.save_bicubic_button.config(state=tk.DISABLED)
            self._clear_crop_views()
        else:
            self.output_image_pil = final_img
            self.bicubic_comparison_pil = bicubic_comp 

            self.original_display_thumb = display_image_in_label(self.bicubic_comparison_pil, self.original_image_label, config.IMAGE_DISPLAY_SIZE)
            self.original_title_label.config(text=f"Comparison (Bicubic x{factor})")

            self.output_display_thumb = display_image_in_label(self.output_image_pil, self.upscaled_image_label, config.IMAGE_DISPLAY_SIZE)
            self.upscaled_title_label.config(text=f"Output ({mode_used} x{factor})")

            self.update_status(f"Upscaling x{factor} ({mode_used}) complete ({elapsed_time:.2f}s). Output: {self.output_image_pil.width}x{self.output_image_pil.height}")
            self.save_button.config(state=tk.NORMAL)
            self.save_bicubic_button.config(state=tk.NORMAL)

        self._set_all_controls_state(tk.NORMAL)


    def start_training_sequence(self):
        if not self.dataset_available:
            messagebox.showerror("Training Error", f"DIV2K dataset not found or unavailable in {config.DATASET_DIR}. Cannot train.")
            return
        if self.training_in_progress:
            messagebox.showwarning("Training", "Training is already in progress.")
            return

        try:
            train_scale = self.train_upscale_factor_var.get()
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            batch_limit = self.batch_limit_var.get()
            lr = self.lr_var.get()

            if train_scale < 2: raise ValueError("Training scale factor must be >= 2.")
            if epochs <= 0: raise ValueError("Epochs must be > 0.")
            if batch_size <= 0: raise ValueError("Batch Size must be > 0.")
            if batch_limit < 0: raise ValueError("Batch Limit must be >= 0.")
            if lr <= 0: raise ValueError("Learning Rate must be > 0.")

        except (tk.TclError, ValueError) as e:
             messagebox.showerror("Invalid Parameter", f"Invalid training parameter value:\n{e}")
             return

        self.current_train_data_hdf5_path = os.path.join(
            config.PREPROCESSED_DATA_DIR,
            config.HDF5_FILENAME_PATTERN.format(train_scale)
        )
        expected_model_basename = config.DEFAULT_MODEL_FILENAME_PATTERN.format(train_scale)
        current_model_path = self.model_path.get()
        current_model_basename = os.path.basename(current_model_path) if current_model_path else ""

        parsed_scale_from_current = parse_scale_from_filename(current_model_basename)
        if not current_model_path or current_model_basename != expected_model_basename or parsed_scale_from_current != train_scale:
             confirmed_save_path = filedialog.asksaveasfilename(
                title=f"Confirm Model Save Path (for x{train_scale})",
                initialdir=os.path.dirname(current_model_path) if current_model_path and os.path.dirname(current_model_path) else config.DATASET_BASE_DIR, # Suggest base dir
                initialfile=expected_model_basename,
                defaultextension=".pth",
                filetypes=[("PyTorch Weights", "*.pth")])

             if not confirmed_save_path:
                 self.update_status("Model save path selection cancelled. Training aborted.", warning=True)
                 return

             selected_basename = os.path.basename(confirmed_save_path)
             parsed_scale_from_selection = parse_scale_from_filename(selected_basename)
             if parsed_scale_from_selection != train_scale:
                  warn_msg = (
                      f"The selected filename '{selected_basename}' does not seem to match the training scale x{train_scale} "
                      f"(it looks like x{parsed_scale_from_selection} or cannot be parsed).\n\n"
                      "Saving here might be confusing or overwrite a different model.\n\n"
                      "Proceed with saving to this path anyway?"
                  )
                  if not messagebox.askyesno("Filename Mismatch", warn_msg):
                      self.update_status("Save path confirmation cancelled due to filename mismatch. Training aborted.", warning=True)
                      return

             model_save_path = confirmed_save_path
             self.model_path.set(model_save_path) 
        else:
             model_save_path = current_model_path 

        self.update_status(f"Checking/Preparing x{train_scale} preprocessed data (HDF5)...")
        self._set_all_controls_state(tk.DISABLED)

        prep_thread = threading.Thread(target=self._run_preprocessing_worker,
                                       args=(model_save_path, train_scale, epochs, batch_size, batch_limit, lr),
                                       daemon=True)
        prep_thread.start()

    def _run_preprocessing_worker(self, model_save_path, train_scale, epochs, batch_size, batch_limit, lr):
        """ Worker thread for running HDF5 preparation """
        is_ready = prepare_training_data_incremental(
            config.DATASET_DIR,
            self.current_train_data_hdf5_path,
            self.update_status,
            scale=train_scale, 
            patch_size=config.DEFAULT_PATCH_SIZE,
            stride=config.DEFAULT_STRIDE,
            chunk_size=config.HDF5_CHUNK_SIZE
        )
        self.after(0, self._confirm_and_start_training, is_ready, model_save_path, train_scale, epochs, batch_size, batch_limit, lr)

    def _confirm_and_start_training(self, preprocessed_data_ready, save_path, train_scale, epochs, batch_size, batch_limit, lr):
        if not preprocessed_data_ready:
            messagebox.showerror("Preprocessing Failed", f"Failed to prepare training data HDF5 file for x{train_scale} at\n{self.current_train_data_hdf5_path}.\nCannot train.")
            self.update_status("Training cancelled due to preprocessing failure.", error=True)
            self._set_all_controls_state(tk.NORMAL) 
            return

        batch_limit_str = f"{batch_limit}" if batch_limit > 0 else "None"
        confirm_msg = (
            f"Preprocessed data for x{train_scale} is ready.\n\n"
            f"Dataset: {os.path.basename(self.current_train_data_hdf5_path)}\n"
            f"Save Model As: {os.path.basename(save_path)}\n\n"
            f"Parameters:\n"
            f"  Scale: x{train_scale}\n"
            f"  Epochs: {epochs}\n"
            f"  Batch Size: {batch_size}\n"
            f"  Batch Limit/Epoch: {batch_limit_str}\n"
            f"  Learning Rate: {lr:.1e}\n"
            f"  Device: {self.device}\n\n"
            f"WARNING: Training can take a long time, especially on CPU.\n\n"
            "Start training?"
        )
        if not messagebox.askyesno(f"Start Training (x{train_scale})", confirm_msg):
            self.update_status("Training cancelled by user.", warning=True)
            self._set_all_controls_state(tk.NORMAL)
            return

        self.training_in_progress = True
        self.update_status(f"Starting training for x{train_scale} (saving to {os.path.basename(save_path)})...")

        train_thread = threading.Thread(target=self._run_training_worker,
                                        args=(save_path, train_scale, epochs, lr, batch_size, batch_limit),
                                        daemon=True)
        train_thread.start()

    def _run_training_worker(self, save_path, train_scale, epochs, lr, batch_size, batch_limit):
        success = train_model(
            hdf5_path=self.current_train_data_hdf5_path,
            save_path=save_path,
            train_scale=train_scale,
            epochs=epochs,
            learning_rate=lr,
            batch_size=batch_size,
            batch_limit=batch_limit,
            device=self.device,
            status_callback=self.update_status 
        )
        self.after(0, self._finish_training, success, save_path)

    def _finish_training(self, success, save_path):
        self.training_in_progress = False
        if success:
            self.update_status(f"Training complete. Model saved: {os.path.basename(save_path)}")
            messagebox.showinfo("Training Complete", f"Training finished successfully.\nModel saved to {save_path}")
            self.model_path.set(save_path)
            self.load_model()
        else:
            messagebox.showerror("Training Failed", "Training process encountered an error. Check status messages and console output.")

        self._set_all_controls_state(tk.NORMAL)


    def _set_training_controls_state(self, state):
        widgets = [
            self.train_button, self.epochs_spinbox, self.batch_size_spinbox,
            self.batch_limit_spinbox, self.lr_spinbox, self.train_scale_spinbox
        ]
        for widget in widgets:
            if widget:
                effective_state = state if (state == tk.NORMAL and self.dataset_available) else tk.DISABLED
                widget.config(state=effective_state)
        if state == tk.NORMAL and self.train_button:
            self._update_train_button_text() 

    def _set_all_controls_state(self, state):
        model_controls = [self.model_entry, self.browse_model_button, self.load_model_button]
        for widget in model_controls:
            if widget: widget.config(state=state)

        self._set_training_controls_state(state)

        if self.load_image_button: self.load_image_button.config(state=state)

        upscale_spinbox_state = state if (state == tk.NORMAL and self.input_image_pil) else tk.DISABLED
        if self.upscale_factor_spinbox: self.upscale_factor_spinbox.config(state=upscale_spinbox_state)

        save_button_state = state if (state == tk.NORMAL and self.output_image_pil) else tk.DISABLED
        if self.save_button: self.save_button.config(state=save_button_state)

        save_bicubic_state = state if (state == tk.NORMAL and self.bicubic_comparison_pil) else tk.DISABLED
        if self.save_bicubic_button: self.save_bicubic_button.config(state=save_bicubic_state)

        if state == tk.NORMAL:
            self._update_upscale_button_state()
        else:
             if self.upscale_button: self.upscale_button.config(state=tk.DISABLED)

        self.update_idletasks()


    def _update_train_button_text(self):
        if not hasattr(self, 'train_button'): return
        try:
            train_scale = self.train_upscale_factor_var.get()
            self.train_button.config(text=f"Train (for x{train_scale})")
        except tk.TclError:
             self.train_button.config(text=f"Train (for x{config.DEFAULT_TRAIN_UPSCALE_FACTOR})")

    def _update_upscale_button_state(self):
        if not hasattr(self, 'upscale_button'): return

        if not self.input_image_pil or self.upscaling_in_progress:
            self.upscale_button.config(text="Upscale Image", state=tk.DISABLED)
            return

        try:
            selected_factor = self.inference_upscale_factor_var.get()
        except tk.TclError:
             selected_factor = config.DEFAULT_INFERENCE_UPSCALE_FACTOR

        can_use_srcnn = (self.model is not None and
                         self.loaded_model_scale is not None and
                         selected_factor == self.loaded_model_scale)

        if can_use_srcnn:
            button_text = f"Upscale x{selected_factor} (SRCNN)"
            button_state = tk.NORMAL
        else:
            method = "Bicubic"
            reason = ""
            if self.model is None: reason = "(No model)"
            elif self.loaded_model_scale is None: reason = "(Model scale ?)"
            elif selected_factor != self.loaded_model_scale: reason = f"(Model=x{self.loaded_model_scale})"
            button_text = f"Upscale x{selected_factor} ({method} {reason})"
            button_state = tk.NORMAL 

        self.upscale_button.config(text=button_text, state=button_state)

    def _clear_crop_views(self):
        if hasattr(self, 'original_crop_label') and self.original_crop_label:
            self.original_crop_label.config(image=self.empty_crop_image, text="")
            self.original_crop_label.image = self.empty_crop_image 
        if hasattr(self, 'upscaled_crop_label') and self.upscaled_crop_label:
            self.upscaled_crop_label.config(image=self.empty_crop_image, text="")
            self.upscaled_crop_label.image = self.empty_crop_image 

    def _handle_mouse_motion(self, event):
        if not self.bicubic_comparison_pil or not self.output_image_pil:
            return
        if not self.original_display_thumb:
             return

        widget = self.original_image_label
        try:
            widget_w = widget.winfo_width()
            widget_h = widget.winfo_height()
            thumb_w, thumb_h = self.original_display_thumb.size

            if thumb_w <= 0 or thumb_h <= 0: return

            pad_x = max(0, (widget_w - thumb_w) / 2)
            pad_y = max(0, (widget_h - thumb_h) / 2)

            mouse_x_rel_thumb = event.x - pad_x
            mouse_y_rel_thumb = event.y - pad_y

            if not (0 <= mouse_x_rel_thumb < thumb_w and 0 <= mouse_y_rel_thumb < thumb_h):
                self._handle_mouse_leave(event)
                return

            full_comp_w, full_comp_h = self.bicubic_comparison_pil.size
            scale_x = full_comp_w / thumb_w
            scale_y = full_comp_h / thumb_h

            full_center_x = mouse_x_rel_thumb * scale_x
            full_center_y = mouse_y_rel_thumb * scale_y

            crop_w, crop_h = config.CROP_SOURCE_SIZE
            crop_x1 = int(full_center_x - crop_w / 2)
            crop_y1 = int(full_center_y - crop_h / 2)
            crop_x1 = max(0, min(crop_x1, full_comp_w - crop_w))
            crop_y1 = max(0, min(crop_y1, full_comp_h - crop_h))
            crop_x2 = crop_x1 + crop_w
            crop_y2 = crop_y1 + crop_h

            crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)

            orig_crop_pil = self.bicubic_comparison_pil.crop(crop_box)
            out_crop_pil = self.output_image_pil.crop(crop_box)

            display_size = config.CROP_DISPLAY_SIZE
            orig_crop_resized = orig_crop_pil.resize(display_size, Image.Resampling.NEAREST)
            out_crop_resized = out_crop_pil.resize(display_size, Image.Resampling.NEAREST)

            orig_crop_tk = ImageTk.PhotoImage(orig_crop_resized)
            out_crop_tk = ImageTk.PhotoImage(out_crop_resized)

            self.original_crop_label.config(image=orig_crop_tk, text="")
            self.original_crop_label.image = orig_crop_tk

            self.upscaled_crop_label.config(image=out_crop_tk, text="")
            self.upscaled_crop_label.image = out_crop_tk

        except Exception as e:
            print(f"Error in mouse motion handler: {e}")


    def _handle_mouse_leave(self, event):
        self._clear_crop_views()


    def on_closing(self):
        print("Closing application...")
        self.destroy()
