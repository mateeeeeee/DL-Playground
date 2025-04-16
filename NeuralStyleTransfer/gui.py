import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os
import queue
from typing import Optional, Tuple, Any
from PIL import Image, ImageTk


try:
    from style_transfer import StyleTransfer, StyleTransferError
except ImportError:
    messagebox.showerror(
        "Import Error",
        "Could not import 'style_transfer.py'. Make sure it's in the same directory."
    )
    exit(1)

PREVIEW_SIZE: Tuple[int, int] = (200, 200)
DEFAULT_PROCESS_SIZE: int = 256
DEFAULT_NUM_STEPS: int = 300
DEFAULT_STYLE_WEIGHT: float = 1e6
DEFAULT_CONTENT_WEIGHT: float = 1.0
APP_TITLE: str = "PyTorch Neural Style Transfer"
GEOMETRY: str = "750x650"
STATUS_READY: str = "Status: Ready"
STATUS_FAILED: str = "Status: Failed. Check console for errors."

class StyleTransferApp(tk.Tk):
    """
    Main application window for the Neural Style Transfer GUI.
    Handles user interaction, image display, parameter settings,
    and communication with the background style transfer process.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        self.title(APP_TITLE)
        self.geometry(GEOMETRY)


        self._setup_variables()
        self._create_widgets()
        self._layout_widgets()
        self._start_queue_checker()

    def _setup_variables(self) -> None:
        """Initialize Tkinter variables and application state."""
        self.style_transfer = StyleTransfer()
        self.content_path = tk.StringVar()
        self.style_path = tk.StringVar()
        self.generated_image_pil: Optional[Image.Image] = None
        self.process_size = tk.IntVar(value=DEFAULT_PROCESS_SIZE)
        self.num_steps = tk.IntVar(value=DEFAULT_NUM_STEPS)
        self.style_weight = tk.DoubleVar(value=DEFAULT_STYLE_WEIGHT)
        self.content_weight = tk.DoubleVar(value=DEFAULT_CONTENT_WEIGHT)
        self.resize_to_original = tk.BooleanVar(value=True)

        self.status_queue: queue.Queue[str] = queue.Queue()
        self.result_queue: queue.Queue[Optional[Image.Image]] = queue.Queue()

        self.transfer_thread: Optional[threading.Thread] = None

    def _create_widgets(self) -> None:
        """Create all the GUI widgets."""

        self.control_frame = ttk.Frame(self, padding="10")
        self.image_frame = ttk.Frame(self, padding="10")
        self.status_frame = ttk.Frame(self, padding=(10, 5))

        self._create_control_widgets(self.control_frame)
        self._create_image_widgets(self.image_frame)

        self.status_label = ttk.Label(self.status_frame, text=STATUS_READY, anchor=tk.W)
        self.save_button = ttk.Button(
            self.status_frame,
            text="Save Result",
            command=self.save_result,
            state=tk.DISABLED
        )

    def _create_control_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets within the control frame."""

        ttk.Label(parent, text="Content Image:").grid(
            row=0, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.content_entry = ttk.Entry(parent, textvariable=self.content_path, width=40)
        self.content_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(
            parent, text="Browse...", command=lambda: self._select_image("content")
        ).grid(row=0, column=2, padx=5, pady=5)


        ttk.Label(parent, text="Style Image:").grid(
            row=1, column=0, padx=5, pady=5, sticky=tk.W
        )
        self.style_entry = ttk.Entry(parent, textvariable=self.style_path, width=40)
        self.style_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(
            parent, text="Browse...", command=lambda: self._select_image("style")
        ).grid(row=1, column=2, padx=5, pady=5)


        settings_frame = ttk.Frame(parent)
        settings_frame.grid(row=2, column=0, columnspan=3, pady=10, sticky=tk.W)
        self._create_settings_widgets(settings_frame)

        self.resize_checkbox = ttk.Checkbutton(
            parent,
            text="Resize Result to Original Content Size on Save",
            variable=self.resize_to_original,
            onvalue=True,
            offvalue=False
        )
        self.resize_checkbox.grid(row=3, column=0, columnspan=3, pady=(0, 5), padx=5, sticky=tk.W)

        self.run_button = ttk.Button(
            parent, text="Run Style Transfer", command=self._start_transfer_thread
        )
        self.run_button.grid(row=4, column=0, columnspan=3, pady=10)

    def _create_settings_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for the settings controls."""
        ttk.Label(parent, text="Size:").pack(side=tk.LEFT, padx=(0,2))
        ttk.Entry(parent, textvariable=self.process_size, width=5).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(parent, text="Steps:").pack(side=tk.LEFT, padx=(0,2))
        ttk.Entry(parent, textvariable=self.num_steps, width=5).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(parent, text="Style W:").pack(side=tk.LEFT, padx=(0,2))
        ttk.Entry(parent, textvariable=self.style_weight, width=8).pack(side=tk.LEFT, padx=(0,10))
        ttk.Label(parent, text="Content W:").pack(side=tk.LEFT, padx=(0,2))
        ttk.Entry(parent, textvariable=self.content_weight, width=5).pack(side=tk.LEFT, padx=(0,10))

    def _create_image_widgets(self, parent: ttk.Frame) -> None:
        """Create widgets for displaying image previews and results."""
        frame_width = PREVIEW_SIZE[0] + 20
        frame_height = PREVIEW_SIZE[1] + 40

        content_prev_frame = ttk.LabelFrame(
            parent, text="Content Preview", width=frame_width, height=frame_height
        )
        content_prev_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.N)
        content_prev_frame.pack_propagate(False)
        self.content_preview_label = ttk.Label(content_prev_frame)
        self.content_preview_label.pack(expand=True)


        style_prev_frame = ttk.LabelFrame(
            parent, text="Style Preview", width=frame_width, height=frame_height
        )
        style_prev_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.N)
        style_prev_frame.pack_propagate(False)
        self.style_preview_label = ttk.Label(style_prev_frame)
        self.style_preview_label.pack(expand=True)

        result_frame = ttk.LabelFrame(
            parent, text="Generated Image", width=frame_width, height=frame_height
        )
        result_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor=tk.N)
        result_frame.pack_propagate(False)
        self.result_image_label = ttk.Label(result_frame)
        self.result_image_label.pack(expand=True)

    def _layout_widgets(self) -> None:
        """Layout the main frames in the window."""
        self.control_frame.pack(side=tk.TOP, fill=tk.X)
        self.control_frame.columnconfigure(1, weight=1)

        img_frame_w = (PREVIEW_SIZE[0] * 3) + 80
        img_frame_h = PREVIEW_SIZE[1] + 60
        self.image_frame.config(width=img_frame_w, height=img_frame_h)
        self.image_frame.pack_propagate(False)
        self.image_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.save_button.pack(side=tk.RIGHT, padx=5)

    def _start_queue_checker(self) -> None:
        """Initiate the periodic checking of the communication queues."""
        self.after(100, self._check_queues)

    def _update_status(self, message: str) -> None:
        """ Safely update the status label from any thread. """
        self.after(0, lambda: self.status_label.config(text=f"Status: {message}"))

    def _select_image(self, image_type: str) -> None:
        """
        Opens file dialog to select an image and updates path and preview.
        """
        file_path = filedialog.askopenfilename(
            title=f"Select {image_type.capitalize()} Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")]
        )
        if not file_path:
            return
        try:
            with Image.open(file_path) as img:
                img.verify()

            if image_type == "content":
                self.content_path.set(file_path)
                self._update_preview(file_path, self.content_preview_label)
            elif image_type == "style":
                self.style_path.set(file_path)
                self._update_preview(file_path, self.style_preview_label)
            self._clear_result()

        except (FileNotFoundError, SyntaxError, Exception) as e:
            messagebox.showerror("Image Load Error", f"Could not load image:\n{file_path}\n\nError: {e}")
            self._update_status(f"Error loading {image_type} image preview.")

    def _update_preview(self, file_path: str, label_widget: ttk.Label) -> None:
        """
        Loads image, resizes for preview, and displays it in the given label.
        """
        try:
            img = Image.open(file_path)
            img.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            label_widget.config(image=photo)

            label_widget.image = photo
        except Exception as e:
            self._update_status(f"Error loading preview: {e}")
            print(f"Error updating preview for {file_path}: {e}")
            label_widget.config(image='')
            label_widget.image = None

    def _clear_result(self) -> None:
        """Clears the result image display and disables the save button."""
        self.result_image_label.config(image='')
        self.result_image_label.image = None
        self.generated_image_pil = None
        self.save_button.config(state=tk.DISABLED)
        if self.status_label.cget("text") not in (STATUS_READY, STATUS_FAILED):
             self._update_status("Ready (result cleared).")


    def _validate_inputs(self) -> bool:
        """Validate selected image paths and settings."""
        content = self.content_path.get()
        style = self.style_path.get()

        if not content or not style:
            messagebox.showwarning("Input Missing", "Please select both Content and Style images.")
            self._update_status("Input Missing: Select both images.")
            return False
        if not os.path.exists(content):
            messagebox.showerror("File Not Found", f"Content image not found:\n{content}")
            self._update_status("Error: Content image file not found.")
            return False
        if not os.path.exists(style):
            messagebox.showerror("File Not Found", f"Style image not found:\n{style}")
            self._update_status("Error: Style image file not found.")
            return False

        try:
            size = self.process_size.get()
            steps = self.num_steps.get()
            sw = self.style_weight.get()
            cw = self.content_weight.get()
            if not (size > 0 and steps > 0 and sw >= 0 and cw >= 0):
                raise ValueError("Settings must be positive (weights >= 0).")
        except (tk.TclError, ValueError) as e:
            messagebox.showerror("Invalid Settings", f"Invalid processing settings:\n{e}")
            self._update_status(f"Error: Invalid settings ({e}).")
            return False

        return True

    def _start_transfer_thread(self) -> None:
        """Validates inputs and starts the style transfer process in a background thread."""
        if not self._validate_inputs():
            return

        if self.transfer_thread and self.transfer_thread.is_alive():
             messagebox.showwarning("Busy", "Style transfer is already running.")
             return

        self._update_status("Starting...")
        self.run_button.config(state=tk.DISABLED)
        self._clear_result()

        content = self.content_path.get()
        style = self.style_path.get()
        size = self.process_size.get()
        steps = self.num_steps.get()
        sw = self.style_weight.get()
        cw = self.content_weight.get()

        self.transfer_thread = threading.Thread(
            target=self.style_transfer.run_style_transfer_task,
            args=(content, style, size, self.status_queue, self.result_queue),
            kwargs={'num_steps': steps, 'style_weight': sw, 'content_weight': cw},
            daemon=True
        )
        self.transfer_thread.start()

    def _check_queues(self) -> None:
        """Checks status and result queues for updates from the background thread."""
        try:
            while True:
                status_msg = self.status_queue.get_nowait()
                self.status_label.config(text=f"Status: {status_msg}")
        except queue.Empty:
            pass

        try:
            result_pil = self.result_queue.get_nowait()
            if isinstance(result_pil, Image.Image):
                self.generated_image_pil = result_pil
                self._update_result_display(result_pil)
                self.save_button.config(state=tk.NORMAL)

                if "Optimizing" in self.status_label.cget("text") or "Step" in self.status_label.cget("text"):
                     self.status_label.config(text="Status: Finished.")
            elif isinstance(result_pil, Exception):
                 error_message = str(result_pil)
                 self.status_label.config(text=f"Status: Error - {error_message}")
                 messagebox.showerror("Style Transfer Error", f"An error occurred during processing:\n\n{error_message}")
                 self._clear_result()
            else:
                if "Error" not in self.status_label.cget("text"):
                     self.status_label.config(text=STATUS_FAILED)
                messagebox.showerror("Style Transfer Failed", "The style transfer process failed. Check console output for details.")
                self._clear_result()

            self.run_button.config(state=tk.NORMAL)
            self.transfer_thread = None

        except queue.Empty:
            pass


        self.after(100, self._check_queues)

    def _update_result_display(self, pil_image: Image.Image) -> None:
        """
        Updates the result image label with the generated image preview.
        """
        try:
            display_image = pil_image.copy()
            display_image.thumbnail(PREVIEW_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(display_image)

            self.result_image_label.config(image=photo)
            self.result_image_label.image = photo
        except Exception as e:
            self._update_status(f"Error displaying result: {e}")
            print(f"Error updating result display: {e}")
            self.result_image_label.config(image='')
            self.result_image_label.image = None

    def save_result(self) -> None:
        """
        Saves the generated image.
        """
        if not self.generated_image_pil:
            messagebox.showwarning("No Image", "No generated image is available to save.")
            self._update_status("Warning: No generated image to save.")
            return

        original_content_path = self.content_path.get()
        default_filename = "stylized_image.png"
        if original_content_path and os.path.exists(original_content_path):
            base, _ = os.path.splitext(os.path.basename(original_content_path))
            default_filename = f"stylized_{base}.png"

        file_path = filedialog.asksaveasfilename(
            title="Save Generated Image",
            initialfile=default_filename,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"),
                       ("JPEG Image", "*.jpg;*.jpeg"),
                       ("Bitmap Image", "*.bmp"),
                       ("TIFF Image", "*.tif;*.tiff"),
                       ("All Files", "*.*")]
        )

        if not file_path:
            self._update_status("Save cancelled.")
            return

        try:
            image_to_save = self.generated_image_pil
            save_message_suffix = f"(at {image_to_save.size[0]}x{image_to_save.size[1]})"

            if self.resize_to_original.get():
                self._update_status("Checking original size...")
                self.update_idletasks()

                original_content_path = self.content_path.get()
                if not original_content_path or not os.path.exists(original_content_path):
                    messagebox.showerror("Upscale Error", "Original content image path is invalid or file missing. Cannot resize.")
                    self._update_status("Error: Original content image path invalid for resizing.")
                    return

                try:
                    with Image.open(original_content_path) as img:
                        original_size = img.size
                except Exception as e:
                    messagebox.showerror("Upscale Error", f"Error reading original content image size:\n{e}")
                    self._update_status(f"Error reading original content size: {e}")
                    print(f"Error reading original content image {original_content_path}: {e}")
                    return

                if self.generated_image_pil.size != original_size:
                    self._update_status("Resizing to original dimensions...")
                    self.update_idletasks()
                    print(f"Resizing result from {self.generated_image_pil.size} to {original_size}...")

                    image_to_save = self.generated_image_pil.resize(original_size, Image.Resampling.LANCZOS)
                    save_message_suffix = f"(resized to {original_size[0]}x{original_size[1]})"
                    print("Resizing complete.")
                else:
                    print("Result already matches original size. No resizing needed.")
                    save_message_suffix = f"(at original size {original_size[0]}x{original_size[1]})"

            self._update_status("Saving image...")
            self.update_idletasks()

            save_dir = os.path.dirname(file_path)
            if save_dir:
                 os.makedirs(save_dir, exist_ok=True)

            image_to_save.save(file_path)
            short_path = os.path.basename(file_path)
            self._update_status(f"Saved {save_message_suffix} to {short_path}")
            print(f"Image saved to {file_path}")

        except Exception as e:
            messagebox.showerror("Save Error", f"An error occurred while saving:\n{e}")
            self._update_status(f"Error during save process: {e}")
            print(f"Error saving image to {file_path}: {e}")