import os
import sys
import json
import time
import requests
import shutil
import logging
import ctypes
from datetime import datetime
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import threading
import warnings
import concurrent.futures
import queue
from urllib.parse import unquote

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_pdf import PdfPages

# --- Path Configuration (anchored to this interface folder) ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(APP_DIR, "logs")
APP_LOG_FILE = os.path.join(LOG_DIR, "debug_app.log")
TEMP_IMAGE_DIR = os.path.join(APP_DIR, "_temp_images")
HISTORY_FILE = os.path.join(APP_DIR, "history.json")
MODEL_CKPT_CANDIDATES = (
    os.path.join(APP_DIR, "multimodal_model.pt"),
    os.path.join(PROJECT_ROOT, "multimodal_model.pt"),
)

os.makedirs(LOG_DIR, exist_ok=True)

# --- Warning & Logging Configuration ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
warnings.filterwarnings("ignore", module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*tight_layout.*")
warnings.filterwarnings("ignore", message=".*QuickGELU.*")
warnings.filterwarnings("ignore", message=".*quick_gelu.*")
# Suppress InsecureRequestWarning from session.verify=False (SSL bypass for broken certs)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

logging.basicConfig(
    filename=APP_LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logging.info("=== Application Started ===")

# --- UI Theme ---
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Shared palette
_C_ROOT   = "#0d1117"
_C_PANEL  = "#161b22"
_C_PANEL2 = "#0f1923"
_C_BORDER = "#1e293b"
_C_ACCENT = "#4f46e5"
_C_BLUE   = "#93c5fd"
_C_TEXT   = "#e2e8f0"
_C_MUTED  = "#64748b"

# Typography constants — populated in FakeNewsApp.__init__() after the
# Tk root window exists (CTkFont requires a root; module-level creation crashes).
_F_TITLE = _F_HEADING = _F_SUBHEAD = _F_BODY = _F_BUTTON = None
_F_METRIC = _F_MONO = _F_LIVE = _F_STATUS = _F_CAPTION = None
_F_RESULT = _F_OVERLAY = _F_TIMER = _F_PROGRESS = None

class FakeNewsApp(ctk.CTk):
    """
    Main application class for the Multimodal Fake News Detector.
    Inherits from customtkinter.CTk to provide a modern, themed UI.
    """
    def __init__(self):
        """
        Initializes the application window, state variables, and UI elements.
        Starts background model loading so the UI remains responsive.
        """
        super().__init__(fg_color=_C_ROOT)

        # Initialise typography system here — CTkFont requires an active Tk root
        global _F_TITLE, _F_HEADING, _F_SUBHEAD, _F_BODY, _F_BUTTON
        global _F_METRIC, _F_MONO, _F_LIVE, _F_STATUS, _F_CAPTION
        global _F_RESULT, _F_OVERLAY, _F_TIMER, _F_PROGRESS
        _F_TITLE    = ctk.CTkFont(family="Segoe UI", size=23, weight="bold")
        _F_HEADING  = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")
        _F_SUBHEAD  = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        _F_BODY     = ctk.CTkFont(family="Segoe UI", size=12)
        _F_BUTTON   = ctk.CTkFont(family="Segoe UI", size=13, weight="bold")
        _F_METRIC   = ctk.CTkFont(family="Consolas", size=13, weight="bold")
        _F_MONO     = ctk.CTkFont(family="Consolas", size=13)
        _F_LIVE     = ctk.CTkFont(family="Segoe UI", size=15, slant="italic")
        _F_STATUS   = ctk.CTkFont(family="Segoe UI", size=11)
        _F_CAPTION  = ctk.CTkFont(family="Segoe UI", size=10)
        _F_RESULT   = ctk.CTkFont(family="Segoe UI", size=28, weight="bold")
        _F_OVERLAY  = ctk.CTkFont(family="Segoe UI", size=24, weight="bold")
        _F_TIMER    = ctk.CTkFont(family="Consolas", size=22, weight="bold")
        _F_PROGRESS = ctk.CTkFont(family="Segoe UI", size=15, weight="bold")

        self.is_destroyed = False
        self.title("📰 Multimodal Fake News Detector")
        self.configure(bg="#0d1117")
        
        # Center Window safely within screen limits
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        window_width = min(1150, screen_width - 160)
        window_height = min(900, screen_height - 160)
        
        x = max(0, int((screen_width / 2) - (window_width / 2)))
        y = max(0, int((screen_height / 2) - (window_height / 2)) - 30)
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.minsize(min(1000, window_width), min(700, window_height))
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # State Management
        self.image_path = None
        self.json_path = None
        self.model = None
        self.tokenizer = None
        self.image_preprocess = None
        self.is_processing = False
        self.cancel_requested = False
        self.start_time = 0
        self.history_file = HISTORY_FILE
        self.log_file = APP_LOG_FILE
        self.json_base_dir = APP_DIR
        self.single_preview_ctk = None
        self.overlay_preview_ctk = None

        # Cache to prevent Tkinter image garbage collection issues.
        self._image_cache = []
        self._after_ids = []

        self.final_tp = self.final_tn = self.final_fp = self.final_fn = 0
        self.batch_results_data = []
        self.report_text = "No analysis report available yet."

        # Thread-local networking session storage (safe per worker).
        self._thread_local = threading.local()
        self.batch_run_id = 0

        # Priority Queue for Producer-Consumer Batch Processing
        self.inference_queue = queue.PriorityQueue()


        self._initialize_temp_directory()
        self._build_ui()
        # Hook Windows drag-and-drop after the window is fully realised
        self.safe_after(200, self._setup_drop_target)
        self.safe_after(100, self._start_model_load_thread)

    def safe_after(self, ms, func, *args):
        """A safe wrapper around Tkinter's 'after' method to prevent shutdown errors."""
        if getattr(self, 'is_destroyed', False):
            return

        def wrapper():
            if getattr(self, 'is_destroyed', False):
                return
            try:
                func(*args)
            except Exception as e:
                # Ignore lingering task execution errors when window is destroyed
                if "invalid command name" not in str(e) and "application has been destroyed" not in str(e):
                    logging.error(f"Background task error: {e}")

        try:
            after_id = self.after(ms, wrapper)
            self._after_ids.append(after_id)
        except Exception:
            pass

    def _initialize_temp_directory(self):
        """
        Creates or clears a temporary directory for storing uploaded or downloaded images.
        """
        self.temp_dir = TEMP_IMAGE_DIR
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logging.info(f"Deleted old {self.temp_dir} directory.")
            except Exception as e:
                logging.error(f"Failed to delete old temp dir: {e}")
                
        os.makedirs(self.temp_dir, exist_ok=True)
        logging.info(f"Created {self.temp_dir} directory.")

    def on_closing(self):
        """
        Handles application shutdown gracefully by cancelling pending tasks and threads.
        """
        logging.info("Application closing sequence initiated.")
        self.is_destroyed = True
        self.is_processing = False
        self.cancel_requested = True

        # Cancel any pending after callbacks to prevent invalid command errors
        for after_id in getattr(self, '_after_ids', []):
            try:
                self.after_cancel(after_id)
            except Exception:
                pass

        # Best-effort queue drain so background worker callbacks don't linger.
        while not self.inference_queue.empty():
            try:
                self.inference_queue.get_nowait()
                self.inference_queue.task_done()
            except queue.Empty:
                break

        # Clean up matplotlib figures to prevent dangling Tkinter updates
        try:
            plt.close('all')
        except Exception:
            pass

        self.quit()  # Stop mainloop first
        self.destroy()

    # ==================== UI CONSTRUCTION ====================
    def _build_ui(self):
        """
        Constructs the main layout: Left Panel (Input) and Right Panel (Output/Dashboard).
        """
        self.grid_columnconfigure(0, weight=3) 
        self.grid_columnconfigure(1, weight=7) 
        self.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            self,
            text="Multimodal Fake News Detector",
            font=_F_TITLE,
            text_color=_C_BLUE,
        ).grid(row=0, column=0, columnspan=2, pady=(14, 8))

        # --- LEFT PANEL ---
        self.input_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.input_frame.grid(row=1, column=0, padx=(16, 8), pady=10, sticky="nsew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_rowconfigure(2, weight=1) 
        
        self._build_single_analysis_block()
        self._build_batch_analysis_block()
        
        self.history_btn = ctk.CTkButton(
            self.input_frame, text="🕒 View History",
            fg_color="#334155", hover_color="#1e293b",
            font=_F_BUTTON, border_width=0, command=self.show_history_window,
        )
        self.history_btn.grid(row=3, column=0, sticky="w", pady=(10, 5))

        self.status_label = ctk.CTkLabel(
            self.input_frame,
            text="Loading models... Please wait.",
            font=_F_STATUS, text_color="gray",
        )
        self.status_label.grid(row=4, column=0, sticky="w")

        self._build_left_panel_overlay()

        # --- RIGHT PANEL ---
        self.output_frame = ctk.CTkFrame(self, fg_color=_C_PANEL2, corner_radius=12, border_width=1, border_color=_C_BORDER)
        self.output_frame.grid(row=1, column=1, padx=(8, 16), pady=10, sticky="nsew")
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_rowconfigure(1, weight=1)

        self.right_header = ctk.CTkLabel(
            self.output_frame,
            text="Analysis Dashboard",
            font=_F_SUBHEAD,
            text_color=_C_BLUE,
        )
        self.right_header.grid(row=0, column=0, pady=(14, 8))

        self.view_container = ctk.CTkFrame(self.output_frame, fg_color="transparent")
        self.view_container.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 15))
        self.view_container.grid_columnconfigure(0, weight=1)
        self.view_container.grid_rowconfigure(0, weight=1)

        self._build_single_view()
        self._build_batch_view()

        # Startup: show the batch dashboard (history chart) in the right pane
        self._update_line_chart()
        self.batch_view.grid(row=0, column=0, sticky="nsew")
        self.right_header.configure(text="Analysis Dashboard \u2014 History")

    def _build_single_analysis_block(self):
        """
        Builds the 'Single Analysis' UI section on the left panel.
        Contains text input, image upload button, and the Analyze button.
        """
        single_frame = ctk.CTkFrame(self.input_frame, fg_color=_C_PANEL, corner_radius=10, border_width=1, border_color=_C_BORDER)
        single_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10), ipadx=12, ipady=8)

        ctk.CTkLabel(single_frame, text="Single Analysis",
                     font=_F_HEADING, text_color=_C_BLUE).pack(anchor="w", pady=(0, 8))
        ctk.CTkLabel(single_frame, text="News Text",
                     font=_F_BODY, text_color=_C_TEXT).pack(anchor="w")
        self.textbox = ctk.CTkTextbox(single_frame, height=80, corner_radius=6,
                                      font=_F_BODY)
        self.textbox.pack(fill="x", pady=(0, 10))
        self.textbox.bind("<KeyRelease>", self._on_textbox_change)

        ctk.CTkLabel(single_frame, text="Associated Image (Optional)",
                     font=_F_BODY, text_color=_C_MUTED).pack(anchor="w")

        self.drop_zone = ctk.CTkFrame(single_frame, fg_color="#0d1f33", corner_radius=8,
                                      cursor="hand2", border_width=1, border_color="#1e3a5f")
        self.drop_zone.pack(fill="x", pady=(0, 5))
        self.drop_zone_label = ctk.CTkLabel(
            self.drop_zone,
            text="🖼  Drag & Drop Image Here\nor click to Browse",
            font=_F_CAPTION, text_color="#475569", justify="center",
        )
        self.drop_zone_label.pack(pady=12)
        self.drop_zone.bind("<Button-1>", lambda e: self.upload_image())
        self.drop_zone_label.bind("<Button-1>", lambda e: self.upload_image())
        self.drop_zone.bind("<Enter>", lambda e: self.drop_zone.configure(fg_color="#0d2a4a", border_color="#2e5f8a"))
        self.drop_zone.bind("<Leave>", lambda e: self.drop_zone.configure(fg_color="#0d1f33", border_color="#1e3a5f"))

        self.img_container = ctk.CTkFrame(single_frame, fg_color="transparent")
        self.img_container.pack(anchor="w", fill="x", pady=(0, 10))
        self.img_label = ctk.CTkLabel(self.img_container, text="No image selected",
                                      font=_F_STATUS, text_color="gray")
        self.img_label.pack(side="left")
        self.remove_img_btn = ctk.CTkButton(
            self.img_container, text="❌", width=30,
            fg_color="transparent", hover_color="#7f1d1d",
            command=self.remove_image,
        )

        self.analyze_btn = ctk.CTkButton(
            single_frame, text="🔍 Analyze Single News",
            height=38, font=_F_BUTTON,
            fg_color="#4f46e5", hover_color="#4338ca",
            command=self.run_single_analysis, state="disabled",
        )
        self.analyze_btn.pack(fill="x", pady=(5, 0))

    def _build_batch_analysis_block(self):
        """
        Builds the 'Batch JSON Analysis' UI section on the left panel.
        Contains JSON upload button and Batch Analyze button.
        """
        batch_frame = ctk.CTkFrame(self.input_frame, fg_color=_C_PANEL, corner_radius=10, border_width=1, border_color=_C_BORDER)
        batch_frame.grid(row=1, column=0, sticky="ew", ipadx=12, ipady=8)

        ctk.CTkLabel(batch_frame, text="Batch JSON Analysis",
                     font=_F_HEADING, text_color=_C_BLUE).pack(anchor="w", pady=(0, 8))

        self.upload_json_btn = ctk.CTkButton(
            batch_frame, text="📂 Upload JSON File",
            fg_color="#334155", hover_color="#1e293b", font=_F_BUTTON,
            command=self.upload_json,
        )
        self.upload_json_btn.pack(anchor="w", pady=(0, 5))

        self.json_container = ctk.CTkFrame(batch_frame, fg_color="transparent")
        self.json_container.pack(anchor="w", fill="x", pady=(0, 10))
        self.json_label = ctk.CTkLabel(self.json_container, text="No JSON selected",
                                       font=_F_STATUS, text_color="gray")
        self.json_label.pack(side="left")
        self.remove_json_btn = ctk.CTkButton(
            self.json_container, text="❌", width=30,
            fg_color="transparent", hover_color="#7f1d1d",
            command=self.remove_json,
        )

        self.batch_analyze_btn = ctk.CTkButton(
            batch_frame, text="📊 Run Batch Analysis",
            height=38, font=_F_BUTTON,
            fg_color="#4f46e5", hover_color="#4338ca",
            command=self.run_batch_analysis, state="disabled",
        )
        self.batch_analyze_btn.pack(fill="x", pady=(5, 0))

    def _build_left_panel_overlay(self):
        """
        Creates a translucent overlay for the left panel during processing.
        """
        self.overlay = ctk.CTkFrame(self.input_frame, fg_color=("gray85", "gray20"), corner_radius=10)
        self.overlay.bind("<Button-1>", lambda e: "break")

        self.overlay_loader = ctk.CTkLabel(self.overlay, text="⚙️ Processing...",
                                           font=_F_OVERLAY)
        self.overlay_loader.place(relx=0.5, rely=0.30, anchor="center")

        self.overlay_timer = ctk.CTkLabel(self.overlay, text="00:00",
                                          font=_F_TIMER, text_color="#3498db")
        self.overlay_timer.place(relx=0.5, rely=0.40, anchor="center")

        self.overlay_img_label = ctk.CTkLabel(self.overlay, text="", corner_radius=10)
        self.overlay_img_label.place(relx=0.5, rely=0.70, anchor="center")

    def _build_single_view(self):
        """
        Builds the UI components for displaying single analysis results.
        """
        self.single_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        self.result_label = ctk.CTkLabel(self.single_view, text="-", font=_F_RESULT)
        self.result_label.pack(pady=40)
        self.conf_label = ctk.CTkLabel(self.single_view, text="Confidence: -",
                                       font=_F_METRIC)
        self.conf_label.pack(pady=5)
        self.warning_label = ctk.CTkLabel(self.single_view, text="", text_color="orange",
                                          font=_F_STATUS, wraplength=400)
        self.warning_label.pack(pady=20)

    def _build_batch_view(self):
        """
        Builds the complex dashboard UI for batch analysis, including graphs and metrics.
        """
        self.batch_view = ctk.CTkFrame(self.view_container, fg_color="transparent")
        self.batch_view.grid_columnconfigure(0, weight=1)
        
        self.batch_view.grid_rowconfigure(0, weight=0) # Headline Box
        self.batch_view.grid_rowconfigure(1, weight=0) # Action Label (Errors/Status)
        self.batch_view.grid_rowconfigure(2, weight=0) # Real-time Metrics Label
        self.batch_view.grid_rowconfigure(3, weight=0) # Summary Textbox (Hidden initially)
        self.batch_view.grid_rowconfigure(4, weight=1) # Scalable Graphs (Pie + Line)
        self.batch_view.grid_rowconfigure(5, weight=0) # Action Buttons (Export/Logs)
        self.batch_view.grid_rowconfigure(6, weight=0) # Stop button
        
        self.realtime_progress_label = ctk.CTkLabel(
            self.batch_view, text="", font=_F_PROGRESS,
        )
        self.realtime_progress_label.grid(row=0, column=0, pady=(0, 3))

        self.live_headline_box = ctk.CTkTextbox(
            self.batch_view, height=55, font=_F_LIVE,
            text_color="#3498db", state="disabled",
        )
        self.live_action_label = ctk.CTkLabel(
            self.batch_view, text="", font=_F_STATUS, text_color="#f39c12",
        )

        # Combined metrics block
        self.metrics_block = ctk.CTkFrame(self.batch_view, fg_color="#0d2137",
                                          corner_radius=8, border_width=1, border_color=_C_BORDER)
        self.realtime_metrics_label = ctk.CTkLabel(
            self.metrics_block,
            text="Acc: —  |  Prec: —  |  Rec: —  |  F1: —",
            font=_F_METRIC, text_color="#93c5fd",
        )
        self.realtime_metrics_label.pack(pady=(7, 2))
        self.expert_count_label = ctk.CTkLabel(
            self.metrics_block,
            text="Expert Review Triggered: 0 items",
            font=_F_METRIC, text_color="#93c5fd",
        )
        self.expert_count_label.pack(pady=(0, 7))

        # Matplotlib — 2-panel: donut left + history line chart right
        # constrained_layout avoids the tight_layout/aspect='equal' warning
        self.fig, (self.ax_pie, self.ax_line) = plt.subplots(
            1, 2, figsize=(9, 4.2), dpi=90,
            gridspec_kw={"width_ratios": [1, 1.6], "wspace": 0.35},
            constrained_layout=True,
        )
        self.fig.patch.set_facecolor('#0d1117')
        self.ax_pie.set_facecolor('#0d1117')
        self.ax_line.set_facecolor('#0d1117')

        self.graph_canvas = FigureCanvasTkAgg(self.fig, master=self.batch_view)
        self.graph_canvas.get_tk_widget().grid(row=4, column=0, sticky="nsew", pady=(5, 10))

        self.pie_labels = ['TP', 'TN', 'FP', 'FN']
        self.pie_colors = ['#22c55e', '#3b82f6', '#ef4444', '#f59e0b']

        # Group Action Buttons side by side
        self.action_frame = ctk.CTkFrame(self.batch_view, fg_color="transparent")
        self.action_frame.grid(row=5, column=0, sticky="ew", pady=(0, 5))
        self.action_frame.grid_columnconfigure(0, weight=1)
        self.action_frame.grid_columnconfigure(1, weight=1)

        self.export_btn = ctk.CTkButton(
            self.action_frame,
            text="💾 Export Report & Graphs",
            height=40, font=_F_BUTTON,
            fg_color="#4f46e5", hover_color="#4338ca",
            command=self.export_results,
        )
        self.export_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.view_logs_btn = ctk.CTkButton(
            self.action_frame,
            text="📄 View Logs",
            height=40, font=_F_BUTTON,
            fg_color="#334155", hover_color="#1e293b",
            command=self.view_logs,
        )
        self.view_logs_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        self.action_frame.grid(row=5, column=0, sticky="ew", pady=(0, 5))
        self.action_frame.lower()

        self.stop_btn = ctk.CTkButton(
            self.batch_view, text="🛑 Stop Analysis",
            fg_color="#7f1d1d", hover_color="#991b1b",
            height=35, font=_F_BUTTON,
            command=self.stop_batch_analysis,
        )
        self.stop_btn.grid(row=6, column=0, sticky="ew", pady=(0, 5))
        self.stop_btn.grid_remove()

    # ==================== BACKGROUND INITIALIZATION ====================
    def _start_model_load_thread(self):
        """Only start the model load thread if the model isn't already loaded."""
        if self.model is not None:
            return
        threading.Thread(target=self.load_model_heavy, daemon=True).start()

    def load_model_heavy(self):
        """
        Background task to load PyTorch and CLIP models into memory/VRAM.
        """
        try:
            import torch
            import open_clip
            from models.multimodal_model import FakeNewsMultimodalModel

            logging.info("Loading PyTorch and OpenCLIP models...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "ViT-B-32"
            
            self.tokenizer = open_clip.get_tokenizer(model_name)
            _, _, self.image_preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
            
            self.model = FakeNewsMultimodalModel(freeze_text_encoder=True, freeze_image_encoder=True)
            try:
                checkpoint_path = next(
                    (path for path in MODEL_CKPT_CANDIDATES if os.path.exists(path)),
                    MODEL_CKPT_CANDIDATES[-1],
                )
                self.model.load_state_dict(
                    torch.load(checkpoint_path, map_location=self.device, weights_only=True)
                )
                status = f"✅ Models Ready (Device: {self.device.upper()})"
                logging.info(f"Model loaded successfully from {checkpoint_path} on {self.device}.")
            except FileNotFoundError:
                status = "⚠️ Models Ready (Warning: untrained weights)"
                logging.warning(
                    "multimodal_model.pt not found in expected paths: "
                    f"{MODEL_CKPT_CANDIDATES[0]} | {MODEL_CKPT_CANDIDATES[1]}"
                )
            
            self.model.to(self.device)
            self.model.eval()
            self.safe_after(0, self._enable_ui, status)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            self.safe_after(0, lambda: self.status_label.configure(text="❌ Error loading models", text_color="#ef4444"))

    def _enable_ui(self, status_text):
        """
        Enables UI buttons after model loading is complete.
        """
        self.status_label.configure(text=status_text, text_color="#22c55e" if "✅" in status_text else "orange")
        self.analyze_btn.configure(state="normal")
        self.batch_analyze_btn.configure(state="normal")

    # ==================== INPUT HANDLING ====================
    def _setup_drop_target(self):
        """
        Registers the app window as a WM_DROPFILES target so that image files
        dragged from Windows Explorer land in _handle_drop.

        Key requirements for stability on 64-bit Windows:
          - Every Win32 function must have explicit restype + argtypes.
          - GetAncestor is used instead of GetParent so we always get the
            real top-level HWND regardless of how CTk nests its widgets.
          - The WNDPROC callback and CallWindowProcW must share the same
            64-bit-safe signature (c_ssize_t return, c_size_t for wParam).

        Browser drag-drop uses OLE IDropTarget (different protocol) so it
        cannot be handled here without COM plumbing.
        """
        if os.name != 'nt':
            return
        try:
            self.update_idletasks()  # ensure HWND exists

            # --- resolve the real root HWND ---------------------------------
            GA_ROOT = 2
            _GetAncestor = ctypes.windll.user32.GetAncestor
            _GetAncestor.restype  = ctypes.c_void_p
            _GetAncestor.argtypes = [ctypes.c_void_p, ctypes.c_uint]

            raw_id = self.winfo_id()
            hwnd   = _GetAncestor(ctypes.c_void_p(raw_id), GA_ROOT) or raw_id

            # --- tell Shell this window accepts dropped files ---------------
            _DragAcceptFiles = ctypes.windll.shell32.DragAcceptFiles
            _DragAcceptFiles.restype  = None
            _DragAcceptFiles.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            _DragAcceptFiles(ctypes.c_void_p(hwnd), True)

            # --- Shell32 helpers -------------------------------------------
            _DragQueryFileW = ctypes.windll.shell32.DragQueryFileW
            _DragQueryFileW.restype  = ctypes.c_uint
            _DragQueryFileW.argtypes = [ctypes.c_void_p, ctypes.c_uint,
                                        ctypes.c_wchar_p, ctypes.c_uint]

            _DragFinish = ctypes.windll.shell32.DragFinish
            _DragFinish.restype  = None
            _DragFinish.argtypes = [ctypes.c_void_p]

            # --- window-subclassing helpers --------------------------------
            GWL_WNDPROC = -4
            WM_DROPFILES = 0x0233

            _GetWindowLongPtr = ctypes.windll.user32.GetWindowLongPtrW
            _GetWindowLongPtr.restype  = ctypes.c_ssize_t
            _GetWindowLongPtr.argtypes = [ctypes.c_void_p, ctypes.c_int]

            _SetWindowLongPtr = ctypes.windll.user32.SetWindowLongPtrW
            _SetWindowLongPtr.restype  = ctypes.c_ssize_t
            _SetWindowLongPtr.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]

            _CallWindowProcW = ctypes.windll.user32.CallWindowProcW
            _CallWindowProcW.restype  = ctypes.c_ssize_t
            _CallWindowProcW.argtypes = [
                ctypes.c_ssize_t,   # lpPrevWndFunc (stored as integer)
                ctypes.c_void_p,    # hWnd
                ctypes.c_uint,      # Msg
                ctypes.c_size_t,    # wParam  (UINT_PTR)
                ctypes.c_ssize_t,   # lParam  (LONG_PTR)
            ]

            # snapshot old proc pointer as plain integer
            old_proc = int(_GetWindowLongPtr(ctypes.c_void_p(hwnd), GWL_WNDPROC))

            # 64-bit safe callback prototype
            WNDPROC = ctypes.WINFUNCTYPE(
                ctypes.c_ssize_t,
                ctypes.c_void_p,   # hWnd
                ctypes.c_uint,     # Msg
                ctypes.c_size_t,   # wParam
                ctypes.c_ssize_t,  # lParam
            )

            def _wndproc(h, msg, wp, lp):
                if msg == WM_DROPFILES:
                    try:
                        hdrop = ctypes.c_void_p(wp)
                        n = _DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
                        for i in range(n):
                            length = _DragQueryFileW(hdrop, i, None, 0)
                            buf = ctypes.create_unicode_buffer(length + 1)
                            _DragQueryFileW(hdrop, i, buf, length + 1)
                            # route to Tk main thread; only handle first image found
                            self.safe_after(0, self._handle_drop, buf.value)
                            break
                        _DragFinish(hdrop)
                    except Exception as inner:
                        logging.warning(f"WM_DROPFILES inner error: {inner}")
                    return 0
                return _CallWindowProcW(old_proc, ctypes.c_void_p(h), msg, wp, lp)

            # Allow WM_DROPFILES through UIPI (needed on Windows Vista+)
            try:
                _ChangeWindowMessageFilterEx = ctypes.windll.user32.ChangeWindowMessageFilterEx
                _ChangeWindowMessageFilterEx.restype  = ctypes.c_bool
                _ChangeWindowMessageFilterEx.argtypes = [
                    ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p
                ]
                MSGFLT_ALLOW = 1
                WM_COPYGLOBALDATA = 0x0049
                _ChangeWindowMessageFilterEx(ctypes.c_void_p(hwnd), WM_DROPFILES,      MSGFLT_ALLOW, None)
                _ChangeWindowMessageFilterEx(ctypes.c_void_p(hwnd), WM_COPYGLOBALDATA, MSGFLT_ALLOW, None)
            except Exception:
                pass  # older Windows — not required

            self._wndproc_ref = WNDPROC(_wndproc)          # keep ref alive!
            new_ptr = ctypes.cast(self._wndproc_ref, ctypes.c_void_p).value
            _SetWindowLongPtr(ctypes.c_void_p(hwnd), GWL_WNDPROC, new_ptr)
            logging.info(f"Drag-and-drop hooked on HWND {hwnd}.")
        except Exception as exc:
            logging.warning(f"Could not set up drag-and-drop: {exc}")

    def _handle_drop(self, path):
        """
        Called (on the main thread) when a file is dropped onto the window.
        Only processes image files; ignores anything else.
        """
        path = path.strip(' \n\r\t"\'{}')
        ext = os.path.splitext(path)[1].lower()
        if ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"):
            self._load_single_image(path)
        else:
            logging.info(f"Dropped file ignored (not an image): {path}")

    def upload_image(self):
        """
        Triggered when user clicks the drop zone.
        Opens a file dialog, reads the image, creates a small preview, and displays it.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if file_path:
            self._load_single_image(file_path)

    def _load_single_image(self, file_path):
        """
        Loads an image of ANY resolution into the preview.
        - Saves a 224x224 normalised JPEG to temp dir for model inference.
        - Shows an aspect-ratio-correct thumbnail (max 120px on longest side) in the UI.
        """
        logging.info(f"Single image loaded: {file_path}")
        try:
            # Open and convert regardless of source bit-depth / colour mode
            img = Image.open(file_path).convert("RGB")

            # --- Normalise for model input (fixed 224x224) ---
            model_img = img.resize((224, 224), Image.Resampling.LANCZOS)
            comp_path = os.path.join(self.temp_dir, "single_upload.jpg")
            model_img.save(comp_path, "JPEG", quality=85, optimize=True)
            self.image_path = comp_path

            # --- Preview thumbnail: keep aspect ratio, cap at 120px --------
            preview_img = img.copy()
            preview_img.thumbnail((120, 120), Image.Resampling.LANCZOS)
            
            new_ctk_img = ctk.CTkImage(
                light_image=preview_img,
                dark_image=preview_img,
                size=(preview_img.width, preview_img.height),
            )
            
            self._image_cache.append(new_ctk_img)
            if len(self._image_cache) > 10:
                self._image_cache.pop(0)

            self.single_preview_ctk = new_ctk_img
            self.img_label.configure(text="", image=new_ctk_img)
            self.img_label.update()  # Force UI refresh
            self.remove_img_btn.pack(side="left", padx=(10, 0))  # Show remove button
            # Update drop zone to confirm the loaded file
            if hasattr(self, 'drop_zone_label'):
                fname = os.path.basename(file_path)
                self.drop_zone_label.configure(text=f"✅  {fname}", text_color="#22c55e")
        except (UnidentifiedImageError, OSError) as exc:
            logging.warning(f"Invalid single image selected: {exc}")
            self.image_path = None
            self.img_label.configure(text="Invalid image selected", image="", text_color="#ef4444")
            self.img_label.update()
            self.single_preview_ctk = None
            self.remove_img_btn.pack_forget()  # Hide remove button
            if hasattr(self, 'drop_zone_label'):
                self.drop_zone_label.configure(text="❌  Invalid image file", text_color="#ef4444")

    def remove_image(self):
        """
        Triggered when user clicks '❌' next to the uploaded image.
        Clears the image path, resets the preview label and restores the drop zone hint.
        """
        self.image_path = None
        self.single_preview_ctk = None
        self.img_label.configure(text="No image selected", image="", text_color="gray")
        self.img_label.update()  # Force UI refresh
        self.remove_img_btn.pack_forget()
        # Restore drop-zone label text
        if hasattr(self, 'drop_zone_label'):
            self.drop_zone_label.configure(text="🖼  Drag & Drop Image Here\nor click to Browse", text_color="#6b7280")
        logging.info("Single image removed by user.")

    def upload_json(self):
        """
        Triggered when user clicks 'Upload JSON File'.
        Opens a file dialog, records the path, and updates the label.
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON Files", "*.json")])
        if file_path:
            logging.info(f"JSON uploaded: {file_path}")
            self.json_path = file_path
            self.json_base_dir = os.path.dirname(os.path.abspath(file_path))
            self.json_label.configure(text=os.path.basename(file_path), text_color="white")
            self.remove_json_btn.pack(side="left", padx=(10, 0)) # Show remove button

    def remove_json(self):
        """
        Triggered when user clicks '❌' next to the uploaded JSON file.
        Clears the json path and resets the label, effectively un-uploading it.
        """
        self.json_path = None
        self.json_base_dir = APP_DIR
        self.json_label.configure(text="No JSON selected", text_color="gray")
        self.remove_json_btn.pack_forget()
        logging.info("JSON file removed by user.")

    # ==================== OVERLAY LOGIC ====================
    def _show_overlay(self):
        """
        Displays the processing overlay on the left panel.
        """
        self.is_processing = True
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.start_time = time.time()
        
        self.overlay_preview_ctk = None
        self.overlay_img_label.configure(image="", text="")
        
        self._update_timer()

    def _hide_overlay(self):
        """
        Hides the processing overlay from the left panel.
        """
        self.is_processing = False
        self.overlay.place_forget()

    def _update_timer(self):
        """
        Updates the timer displayed on the processing overlay.
        """
        if not self.is_processing: return
        elapsed = int(time.time() - self.start_time)
        m, s = divmod(elapsed, 60)
        self.overlay_timer.configure(text=f"{m:02d}:{s:02d}")
        self.safe_after(1000, self._update_timer)

    def _update_translucent_image(self, pil_img):
        if pil_img:
            img = pil_img.copy()
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            img = img.resize((240, 240), Image.Resampling.LANCZOS)
            alpha = img.split()[3]
            alpha = alpha.point(lambda p: p * 0.35)
            img.putalpha(alpha)
            
            new_ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(240, 240))
            
            self._image_cache.append(new_ctk_img)
            if len(self._image_cache) > 10:
                self._image_cache.pop(0)
                
            self.overlay_preview_ctk = new_ctk_img
            self.overlay_img_label.configure(image=new_ctk_img, text="")
        else:
            self.overlay_preview_ctk = None
            self.overlay_img_label.configure(image="", text="⚠️ Image Unavailable", text_color="#ef4444")

    def _update_live_headline(self, text):
        """
        Updates the headline text box during live batch processing.
        """
        self.live_headline_box.configure(state="normal")
        self.live_headline_box.delete("1.0", "end")
        self.live_headline_box.insert("1.0", f'"{text}"')
        self.live_headline_box.configure(state="disabled")

    # ==================== CORE ML INFERENCE ====================
    def _get_probabilities(self, text, img_path):
        """
        Runs text and image through the loaded model to predict Real vs Fake.
        Returns (probability_real, probability_fake).
        """
        import torch

        tokens = self.tokenizer([text])[0].unsqueeze(0).to(self.device)

        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as pil_img:
                image_proc = pil_img.convert("RGB")
        else:
            image_proc = Image.new("RGB", (224, 224), (0, 0, 0))

        image_tensor = self.image_preprocess(image_proc).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _, _, _ = self.model(tokens, image_tensor)
            probs = torch.softmax(logits, dim=1).squeeze()

        return float(probs[0].item()), float(probs[1].item())


    # ==================== SINGLE ANALYSIS PIPELINE ====================
    def run_single_analysis(self):
        """
        Triggered when 'Analyze Single News' is clicked.
        Validates input and starts the inference thread.
        """
        text_input = self.textbox.get("1.0", "end").strip()
        if not text_input:
            self.status_label.configure(text="⚠️ Please enter news text first.", text_color="orange")
            return

        logging.info("Starting single analysis.")
        self.batch_view.grid_forget()
        self.single_view.grid(row=0, column=0, sticky="nsew")
        self.right_header.configure(text="Analysis Processing...")
        self._show_overlay()
        
        threading.Thread(target=self._single_inference_task, args=(text_input, self.image_path), daemon=True).start()

    def _single_inference_task(self, text_input, img_path):
        """
        Background thread task for single inference to prevent UI freezing.
        """
        try:
            prob_real, prob_fake = self._get_probabilities(text_input, img_path)
            logging.info(f"Single analysis complete. Real: {prob_real:.2f}, Fake: {prob_fake:.2f}")
            self.safe_after(0, self._update_single_results, prob_real, prob_fake)
        except Exception as e:
            logging.error(f"Single inference error: {e}")
            self.safe_after(0, lambda: self.right_header.configure(text="Analysis Results"))
            self.safe_after(0, self._hide_overlay)

    def _on_textbox_change(self, event=None):
        """Clears the status warning whenever the user types in the news textbox."""
        current = self.status_label.cget("text")
        if "⚠️" in current and "text" in current.lower():
            # Restore to model-ready status if model is loaded, else blank
            if self.model is not None:
                self.status_label.configure(
                    text=f"✅ Models Ready (Device: {getattr(self, 'device', 'cpu').upper()})",
                    text_color="#22c55e"
                )
            else:
                self.status_label.configure(text="", text_color="gray")

    def _update_single_results(self, prob_real, prob_fake):
        """
        Updates the UI with the final result of the single analysis.
        """
        self._hide_overlay()
        self.right_header.configure(text="Analysis Results")
        max_conf = max(prob_real, prob_fake) * 100
        
        if prob_fake > prob_real:
            self.result_label.configure(text="🚨 FAKE NEWS", text_color="#ef4444")
        else:
            self.result_label.configure(text="✅ REAL NEWS", text_color="#22c55e")
            
        self.conf_label.configure(text=f"Confidence: {max_conf:.2f}%")
        self.warning_label.configure(text=f"⚠️ Low confidence. Expert review recommended." if max_conf < 65.0 else "")

    # ==================== BATCH PRODUCER-CONSUMER PIPELINE ====================
    def run_batch_analysis(self):
        """
        Triggered when 'Run Batch Analysis' is clicked.
        Sets up the UI dashboard and starts the batch manager thread.
        """
        if self.model is None or self.tokenizer is None or self.image_preprocess is None:
            self.status_label.configure(text="Model not ready yet. Please wait.", text_color="orange")
            return

        if not self.json_path:
            self.status_label.configure(text="Please upload a JSON file.", text_color="orange")
            return

        logging.info(f"Starting batch analysis: {self.json_path}")
        self.single_view.grid_forget()
        self.batch_view.grid(row=0, column=0, sticky="nsew")
        self.right_header.configure(text="Batch Processing Pipeline...")

        self.cancel_requested = False
        self.action_frame.lower()       # hide without removing from grid
        self.metrics_block.grid_remove()

        # Show only the donut during processing — hide history line chart
        self.ax_line.set_visible(False)
        self.ax_pie.set_visible(True)
        self.ax_pie.set_position([0.08, 0.10, 0.84, 0.80])  # full-width donut
        self.ax_pie.clear()
        self.ax_pie.set_facecolor('#0d1117')
        self.ax_pie.set_aspect('equal')
        self.ax_pie.pie([1], colors=['#161b22'], radius=1,
                        wedgeprops=dict(width=0.46, edgecolor='#0d1117', linewidth=2))
        self.ax_pie.text(0, 0, "Starting...", ha='center', va='center',
                         color='#334155', fontsize=13, weight='bold')
        self.ax_pie.set_title("Live Prediction Distribution",
                              color='#93c5fd', pad=12, weight='bold', fontsize=11)
        self.graph_canvas.draw_idle()

        self.stop_btn.configure(state="normal", text="🛑 Stop Analysis")
        self.stop_btn.grid()

        self.live_headline_box.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.live_action_label.grid(row=1, column=0, pady=(0, 4))
        self.metrics_block.grid(row=2, column=0, sticky="ew", pady=(0, 6), padx=4)
        self.realtime_metrics_label.configure(text="Acc: —  |  Prec: —  |  Rec: —  |  F1: —")
        self.expert_count_label.configure(text="Expert Review Triggered: 0 items")
        self.realtime_progress_label.configure(text="Preparing batch...")

        self.inference_queue = queue.PriorityQueue()
        self.batch_run_id += 1
        current_run_id = self.batch_run_id
        self._show_overlay()

        threading.Thread(
            target=self._batch_manager_thread,
            args=(current_run_id,),
            daemon=True,
        ).start()

    def stop_batch_analysis(self):
        """
        Interrupts an ongoing batch analysis.
        """
        self.cancel_requested = True
        self.live_action_label.configure(text="🛑 Stopping... finishing current items.", text_color="#ef4444")
        self.stop_btn.configure(state="disabled", text="Stopping...")
        logging.info("User requested to stop batch analysis mid-way.")

    def _get_thread_session(self):
        """
        Provides a thread-local requests session for concurrent image downloading.
        """
        if not hasattr(self._thread_local, "session"):
            import ssl
            session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=8,
                pool_maxsize=8,
                max_retries=2,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update(
                {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            )
            # Disable SSL verification to handle self-signed / misconfigured certs
            session.verify = False
            self._thread_local.session = session
        return self._thread_local.session

    def _batch_manager_thread(self, run_id):
        """Manager: submit producer tasks, then signal consumer completion safely."""
        submitted_count = 0
        consumer_started = False
        futures = []
        executor = None

        try:
            if run_id != self.batch_run_id:
                return

            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("Batch JSON root must be a list of items.")

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
            for idx, item in enumerate(data):
                if self.cancel_requested or run_id != self.batch_run_id:
                    break
                futures.append(executor.submit(self._download_producer_task, item, idx, run_id))

            submitted_count = len(futures)

            if submitted_count == 0:
                self.safe_after(0, lambda: self.realtime_progress_label.configure(text="No items to process."))
                self.safe_after(0, self._finalize_batch, 0, 0, 0, 0, 0, 0)
                return

            self.safe_after(0, lambda: self.realtime_progress_label.configure(text=f"Processing 0/{submitted_count}"))
            threading.Thread(
                target=self._inference_consumer_thread,
                args=(submitted_count, run_id),
                daemon=True,
            ).start()
            consumer_started = True

            for future in concurrent.futures.as_completed(futures):
                if self.cancel_requested or run_id != self.batch_run_id:
                    break
                try:
                    future.result()
                except Exception:
                    logging.exception("Producer task crashed unexpectedly.")

        except Exception as e:
            logging.error(f"Batch Manager error: {e}")
            if run_id == self.batch_run_id:
                self.safe_after(0, self._hide_overlay)
                self.safe_after(0, self.stop_btn.grid_remove)
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

            # Send completion marker only if consumer was actually started
            if consumer_started:
                self.inference_queue.put(((99, 10**12), {"_done": True, "run_id": run_id}))

    def _resolve_local_image_path(self, image_ref):
        """
        Resolves relative or absolute local image paths from JSON references.
        """
        if image_ref is None:
            return None

        candidate = str(image_ref).strip()
        if not candidate:
            return None

        if candidate.lower().startswith("file://"):
            candidate = unquote(candidate[7:])
            if os.name == "nt" and len(candidate) >= 3 and candidate[0] == "/" and candidate[2] == ":":
                candidate = candidate[1:]

        candidate = os.path.expanduser(candidate)

        if os.path.isabs(candidate):
            absolute_candidate = os.path.abspath(candidate)
            return absolute_candidate if os.path.isfile(absolute_candidate) else None

        base_dir = getattr(self, "json_base_dir", APP_DIR)
        relative_to_json = os.path.abspath(os.path.join(base_dir, candidate))
        if os.path.isfile(relative_to_json):
            return relative_to_json

        relative_to_cwd = os.path.abspath(candidate)
        if os.path.isfile(relative_to_cwd):
            return relative_to_cwd

        return None

    def _save_processed_image(self, image_obj, output_path):
        processed = image_obj.convert("RGB").resize((224, 224), Image.Resampling.LANCZOS)
        processed.save(output_path, "JPEG", quality=40, optimize=True)
        return processed.copy()

    def _download_producer_task(self, item, idx, run_id):
        """Producer: download/resolve image and enqueue one payload per item."""
        if not self.is_processing or self.cancel_requested or run_id != self.batch_run_id:
            return

        item_dict = item if isinstance(item, dict) else {"id": "unknown", "text": "", "label": 0}
        item_id = str(item_dict.get("id", "unknown"))
        safe_item_id = item_id.replace("\\", "_").replace("/", "_").replace(":", "_")

        queue_payload = {
            "item": item_dict,
            "img_path": None,
            "display_img": None,
            "error_msg": None,
            "run_id": run_id,
        }

        try:
            if not isinstance(item, dict):
                queue_payload["error_msg"] = "[Schema Error] Item is not a JSON object."
            else:
                image_ref = next(
                    (
                        value for value in (
                            item_dict.get("image_url"),
                            item_dict.get("image_path"),
                            item_dict.get("image"),
                        )
                        if isinstance(value, str) and value.strip()
                    ),
                    "",
                ).strip()

                if not image_ref:
                    queue_payload["error_msg"] = "[Skipped] No image reference found."

                elif image_ref.lower().startswith(("http://", "https://")):
                    img_path = os.path.join(self.temp_dir, f"{safe_item_id}.jpg")
                    session = self._get_thread_session()

                    for attempt in range(4):
                        if self.cancel_requested or run_id != self.batch_run_id:
                            break

                        response = session.get(image_ref, timeout=(3.5, 30.0))
                        if response.status_code in [429, 503]:
                            wait_time = 3 * (2 ** attempt)
                            logging.warning(
                                f"HTTP {response.status_code} for {item_id}. Retrying in {wait_time}s..."
                            )
                            time.sleep(wait_time)
                            continue

                        response.raise_for_status()
                        if len(response.content) < 500:
                            raise ValueError("Payload too small (broken link)")

                        with Image.open(BytesIO(response.content)) as img_raw:
                            queue_payload["display_img"] = self._save_processed_image(img_raw, img_path)
                        queue_payload["img_path"] = img_path
                        break

                    if (
                        queue_payload["img_path"] is None
                        and queue_payload["error_msg"] is None
                        and not self.cancel_requested
                        and run_id == self.batch_run_id
                    ):
                        queue_payload["error_msg"] = "[Network/File Error] Image unavailable"

                else:
                    local_path = self._resolve_local_image_path(image_ref)
                    if local_path is None:
                        queue_payload["error_msg"] = "[Path Error] Local image not found"
                        logging.warning(f"Local image path not found for {item_id}: {image_ref}")
                    else:
                        img_path = os.path.join(self.temp_dir, f"{safe_item_id}.jpg")
                        with Image.open(local_path) as img_raw:
                            queue_payload["display_img"] = self._save_processed_image(img_raw, img_path)
                        queue_payload["img_path"] = img_path

        except Exception as e:
            queue_payload["img_path"] = None
            queue_payload["display_img"] = None
            queue_payload["error_msg"] = "[Network/File Error] Image unavailable"
            logging.warning(f"Download/load failed for {item_id}: {e}")

        if not self.cancel_requested and run_id == self.batch_run_id:
            priority_level = 0 if queue_payload["img_path"] else 1
            self.inference_queue.put(((priority_level, idx), queue_payload))

    def _inference_consumer_thread(self, total_items, run_id):
        """Consumer: sequential inference; exits on manager completion marker."""
        tp = tn = fp = fn = expert = 0
        self.batch_results_data = []
        processed_count = 0

        while True:
            try:
                _, payload = self.inference_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if payload.get("run_id") != run_id:
                self.inference_queue.task_done()
                continue

            if payload.get("_done"):
                self.inference_queue.task_done()
                break

            processed_count += 1
            item = payload.get("item", {})
            text = str(item.get("text", ""))

            raw_label = item.get("label", 0)
            try:
                true_label = int(raw_label)
            except (TypeError, ValueError):
                true_label = 0

            self.safe_after(0, lambda c=processed_count, t=total_items: self.realtime_progress_label.configure(
                text=f"Processing {c}/{t}"
            ))
            self.safe_after(0, lambda t_ext=text: self._update_live_headline(t_ext))

            if payload.get("error_msg"):
                self.safe_after(0, lambda e=payload["error_msg"]: self.live_action_label.configure(text=e, text_color="#ef4444"))
                self.safe_after(0, lambda: self._update_translucent_image(None))
            else:
                self.safe_after(0, lambda: self.live_action_label.configure(text="[Success] Image Optimized & Loaded", text_color="#22c55e"))
                self.safe_after(0, lambda p=payload.get("display_img"): self._update_translucent_image(p))

            try:
                prob_real, prob_fake = self._get_probabilities(text, payload.get("img_path"))
                pred = 1 if prob_fake > prob_real else 0
                confidence = max(prob_real, prob_fake)

                if pred == 1 and true_label == 1:
                    tp += 1
                elif pred == 0 and true_label == 0:
                    tn += 1
                elif pred == 1 and true_label == 0:
                    fp += 1
                elif pred == 0 and true_label == 1:
                    fn += 1

                if confidence < 0.65:
                    expert += 1

                self.batch_results_data.append({
                    "id": item.get("id"),
                    "text": text,
                    "image_url": item.get("image_url") or item.get("image_path") or item.get("image") or "",
                    "img_path": payload.get("img_path"),
                    "predicted": pred,
                    "predicted_label": "Fake" if pred == 1 else "Real",
                    "true": true_label,
                    "true_label": "Fake" if true_label == 1 else "Real",
                    "confidence": round(confidence, 4),
                    "expert_review": confidence < 0.65,
                })

                correct = tp + tn
                total_so_far = correct + fp + fn
                acc = correct / total_so_far if total_so_far > 0 else 0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

                self.safe_after(0, lambda a=acc, p=prec, r=rec, f=f1, ex=expert: [
                    self.realtime_metrics_label.configure(
                        text=f"Acc: {a:.4f}  |  Prec: {p:.4f}  |  Rec: {r:.4f}  |  F1: {f:.4f}"
                    ),
                    self.expert_count_label.configure(
                        text=f"Expert Review Triggered: {ex} items"
                    ),
                ])
                self.safe_after(0, self._fast_update_pie_chart, tp, tn, fp, fn, processed_count, total_items)

            except Exception as e:
                logging.error(f"Inference error for item {item.get('id')}: {e}")

            self.inference_queue.task_done()

        if self.is_processing and run_id == self.batch_run_id:
            self.safe_after(0, self._finalize_batch, tp, tn, fp, fn, expert, processed_count)

    def _fast_update_pie_chart(self, tp, tn, fp, fn, current, total):
        """
        Updates the live donut pie chart in the batch dashboard.
        """
        self.ax_pie.clear()
        # Restore full-width position lost by clear() during processing
        self.ax_pie.set_position([0.08, 0.10, 0.84, 0.80])
        self.ax_pie.set_facecolor('#111827')
        self.ax_pie.set_aspect('equal')

        vals   = [tp, tn, fp, fn]
        labels = self.pie_labels
        colors = self.pie_colors

        fv, fl, fc = zip(*[(v, l, c) for v, l, c in zip(vals, labels, colors) if v > 0]) if any(v > 0 for v in vals) else ([], [], [])

        if not fv:
            self.ax_pie.pie([1], colors=['#1f2937'], radius=1,
                            wedgeprops=dict(width=0.45, edgecolor='#111827'))
            self.ax_pie.text(0, 0, "Ready", ha='center', va='center',
                             color='#6b7280', fontsize=14, weight='bold')
        else:
            wedges, texts, autotexts = self.ax_pie.pie(
                fv, labels=fl, colors=fc,
                autopct='%1.1f%%',
                textprops={'color': 'white', 'fontsize': 9.5, 'weight': 'bold'},
                radius=1,
                wedgeprops=dict(width=0.45, edgecolor='#111827', linewidth=1.5),
                startangle=90,
            )
            for at in autotexts:
                at.set_fontsize(8.5)

        self.ax_pie.text(0, 0, f"{current}\n/{total}",
                         ha='center', va='center',
                         color='white', fontsize=13, weight='bold')
        self.ax_pie.set_title("Live Prediction Distribution",
                              color='#93c5fd', pad=12, weight='bold', fontsize=11)
        # Use subplots_adjust instead of tight_layout (avoids aspect='equal' warning)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05)
        self.graph_canvas.draw_idle()

    def _finalize_batch(self, tp, tn, fp, fn, expert, total):
        """
        Called when batch processing finishes or is stopped.
        Calculates final metrics and saves to history.
        """
        self._hide_overlay()
        logging.info("Batch processing complete.")

        self.final_tp, self.final_tn, self.final_fp, self.final_fn = tp, tn, fp, fn

        # Calculate metrics FIRST (fixes NameError: expert_ratio used before assignment)
        correct       = tp + tn
        accuracy      = correct / total       if total        > 0 else 0
        precision     = tp / (tp + fp)        if (tp + fp)   > 0 else 0
        recall        = tp / (tp + fn)        if (tp + fn)   > 0 else 0
        f1            = (2*precision*recall)  / (precision + recall) if (precision + recall) > 0 else 0
        expert_ratio  = expert / total        if total        > 0 else 0

        self.right_header.configure(text="Batch Analysis Results")
        if self.cancel_requested:
            self.realtime_progress_label.configure(text=f"⚠️ Processing Stopped ({total} items processed).")
        else:
            self.realtime_progress_label.configure(text=f"✅ Processing Complete — {total} items.")

        self.stop_btn.grid_remove()
        self.live_headline_box.grid_remove()
        self.live_action_label.grid_remove()

        # Update the combined metrics block with final values
        self.realtime_metrics_label.configure(
            text=f"Acc: {accuracy:.4f}  |  Prec: {precision:.4f}  |  Rec: {recall:.4f}  |  F1: {f1:.4f}"
        )
        self.expert_count_label.configure(
            text=f"Expert Review Triggered: {expert} items  ({expert_ratio:.1%})"
        )
        self.metrics_block.grid(row=3, column=0, sticky="ew", pady=(0, 6), padx=4)
        self.action_frame.lift()        # reveal instantly — no layout shift
        # Restore both axes now that results are ready
        self.ax_line.set_visible(True)

        raw_time = datetime.now()
        time_str = raw_time.strftime("%I:%M %p").lower().replace("am", "a.m.").replace("pm", "p.m.")
        date_str = raw_time.strftime("%d %b %Y")
        record = {
            "time": time_str,
            "date": date_str,
            "metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "expert_ratio": expert_ratio,
                "tp": tp, "tn": tn, "fp": fp, "fn": fn
            }
        }
        self.save_to_history(record)
        
        self._update_line_chart()

        metrics_text = f"""Confusion Matrix Summary:
---------------------------------------------
True Positive  (Fake -> Fake) : {tp}
True Negative  (Real -> Real) : {tn}
False Positive (Real -> Fake) : {fp}
False Negative (Fake -> Real) : {fn}

Performance Metrics:
---------------------------------------------
Accuracy : {accuracy:.4f}    Precision: {precision:.4f}
Recall   : {recall:.4f}    F1 Score : {f1:.4f}

Expert Review Triggered : {expert} items ({expert_ratio:.2%})
"""
        self.report_text = metrics_text

    # ==================== HISTORICAL GRAPH & DATA ====================
    def _update_line_chart(self):
        """
        Redraws both panels:
          LEFT  — Live donut pie (Ready placeholder or real-time results)
          RIGHT — Historical metrics line chart (Acc / Prec / Rec / F1 per run)
        """
        history = self.load_history()

        BG      = '#0d1117'
        ACC_C   = '#22c55e'
        PREC_C  = '#3b82f6'
        REC_C   = '#f59e0b'
        F1_C    = '#a78bfa'
        GRID_C  = '#1e293b'
        TEXT_C  = '#94a3b8'
        TITLE_C = '#93c5fd'

        no_results = (self.final_tp + self.final_tn + self.final_fp + self.final_fn) == 0

        # ── LEFT: Donut Pie ──────────────────────────────────────────────────
        self.ax_pie.clear()
        self.ax_pie.set_facecolor(BG)

        if no_results:
            # Hide pie axis, expand line chart to fill full figure width
            self.ax_pie.set_visible(False)
            self.ax_line.set_position([0.07, 0.12, 0.90, 0.76])
        else:
            # Restore split layout and draw the donut
            self.ax_pie.set_visible(True)
            self.ax_pie.set_aspect('equal')
            self.ax_pie.set_position([0.06, 0.10, 0.34, 0.76])
            self.ax_line.set_position([0.50, 0.12, 0.46, 0.76])

            tp, tn = self.final_tp, self.final_tn
            fp, fn = self.final_fp, self.final_fn
            vals = [v for v in [tp, tn, fp, fn] if v > 0]
            lbls = [l for v, l in zip([tp, tn, fp, fn], ['TP', 'TN', 'FP', 'FN']) if v > 0]
            clrs = [c for v, c in zip([tp, tn, fp, fn],
                                       [ACC_C, PREC_C, '#ef4444', REC_C]) if v > 0]
            total = tp + tn + fp + fn
            if vals:
                _, _, ats = self.ax_pie.pie(
                    vals, labels=lbls, colors=clrs, autopct='%1.1f%%',
                    textprops={'color': TEXT_C, 'fontsize': 8.5, 'fontweight': 'bold'},
                    radius=1,
                    wedgeprops=dict(width=0.46, edgecolor=BG, linewidth=2),
                    startangle=90,
                )
                for at in ats: at.set_fontsize(7.5)
                self.ax_pie.text(0, 0, f"{total}\nitems", ha='center', va='center',
                                 color='#e2e8f0', fontsize=9, weight='bold')
            self.ax_pie.set_title("Prediction Split", color=TITLE_C,
                                  fontsize=10, pad=10, weight='bold')

        # ── RIGHT: History Line Chart ────────────────────────────────────────────
        self.ax_line.clear()
        self.ax_line.set_facecolor(BG)

        if len(history) < 1:
            # Empty state
            self.ax_line.text(0.5, 0.5, "No history yet\nRun a batch analysis to see trends",
                              ha='center', va='center', transform=self.ax_line.transAxes,
                              color='#334155', fontsize=10, style='italic', linespacing=1.8)
            self.ax_line.set_title("Run History — Metrics Trend", color=TITLE_C,
                                   fontsize=10, pad=10, weight='bold')
        else:
            runs  = list(range(1, len(history) + 1))
            accs  = [h.get('metrics', {}).get('accuracy',  0) for h in history]
            precs = [h.get('metrics', {}).get('precision', 0) for h in history]
            recs  = [h.get('metrics', {}).get('recall',    0) for h in history]
            f1s   = [h.get('metrics', {}).get('f1',        0) for h in history]

            def _plot_line(ax, x, y, color, label):
                ax.plot(x, y, color=color, linewidth=2, label=label,
                        marker='o', markersize=5, markerfacecolor=BG,
                        markeredgecolor=color, markeredgewidth=1.8,
                        solid_capstyle='round', solid_joinstyle='round')
                # Glow effect — soft wide transparent line behind
                ax.plot(x, y, color=color, linewidth=6, alpha=0.12, zorder=1)

            _plot_line(self.ax_line, runs, accs,  ACC_C,  'Accuracy')
            _plot_line(self.ax_line, runs, precs, PREC_C, 'Precision')
            _plot_line(self.ax_line, runs, recs,  REC_C,  'Recall')
            _plot_line(self.ax_line, runs, f1s,   F1_C,   'F1 Score')

            self.ax_line.set_xlim(0.5, max(runs) + 0.5)
            self.ax_line.set_ylim(-0.05, 1.08)
            self.ax_line.set_xticks(runs)
            self.ax_line.set_xticklabels([f"#{r}" for r in runs],
                                          color=TEXT_C, fontsize=8)
            self.ax_line.yaxis.set_tick_params(labelcolor=TEXT_C, labelsize=8)

            # Grid
            self.ax_line.set_axisbelow(True)
            self.ax_line.grid(axis='y', color=GRID_C, linewidth=0.8, linestyle='--', alpha=0.7)
            self.ax_line.grid(axis='x', color=GRID_C, linewidth=0.4, linestyle=':', alpha=0.4)

            # Spines
            for sp in ['top', 'right']:
                self.ax_line.spines[sp].set_visible(False)
            for sp in ['left', 'bottom']:
                self.ax_line.spines[sp].set_edgecolor(GRID_C)
                self.ax_line.spines[sp].set_linewidth(0.8)

            # Legend
            leg = self.ax_line.legend(
                loc='lower right', fontsize=8,
                facecolor='#161b22', edgecolor=GRID_C,
                labelcolor=TEXT_C, framealpha=0.9,
                handlelength=1.6, handletextpad=0.5,
            )
            for line in leg.get_lines():
                line.set_linewidth(2)

            # Latest value annotations on right edge
            for val, color in zip([accs[-1], precs[-1], recs[-1], f1s[-1]],
                                  [ACC_C, PREC_C, REC_C, F1_C]):
                self.ax_line.annotate(
                    f'{val:.2f}',
                    xy=(runs[-1], val),
                    xytext=(4, 0), textcoords='offset points',
                    color=color, fontsize=7.5, fontweight='bold', va='center',
                )

            self.ax_line.set_title("Run History — Metrics Trend", color=TITLE_C,
                                   fontsize=10, pad=10, weight='bold')

        # Don't call tight_layout — we manage positions manually via set_position()
        # (tight_layout is incompatible with axes that have aspect='equal')
        self.graph_canvas.draw_idle()

    def load_history(self):
        """
        Loads previous batch run metrics from the local JSON history file.
        """
        if os.path.exists(self.history_file) and os.path.getsize(self.history_file) > 0:
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data if isinstance(data, list) else []
            except (OSError, json.JSONDecodeError) as exc:
                logging.warning(f"Failed to read history file: {exc}")
        return []

    def save_to_history(self, record):
        """
        Appends a new batch run record to the local JSON history file.
        """
        hist = self.load_history()
        hist.append(record)
        temp_path = f"{self.history_file}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=4)
            os.replace(temp_path, self.history_file)
        except OSError as exc:
            logging.warning(f"Failed to save history: {exc}")
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def show_history_window(self):
        """
        Opens a new window displaying textual history of past batch runs.
        """
        hist = self.load_history()[-10:] 
        if not hist:
            messagebox.showinfo("History", "No history available yet.")
            return

        hist_win = ctk.CTkToplevel(self)
        hist_win.title("Run History — Last 10 Executions")
        hist_win.geometry("640x520")
        hist_win.attributes('-topmost', 'true')
        hist_win.grid_columnconfigure(0, weight=1)
        hist_win.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            hist_win,
            text="📊 Batch Run History",
            font=_F_HEADING,
            text_color="#93c5fd",
        ).grid(row=0, column=0, pady=(14, 6), padx=16, sticky="w")

        textbox = ctk.CTkTextbox(
            hist_win,
            font=_F_MONO,
            text_color="#e2e8f0",
            fg_color="#111827",
            corner_radius=8,
        )
        textbox.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))

        for i, h in enumerate(reversed(hist)):
            if 'time' in h and 'date' in h:
                t_display = f"{h['date']} at {h['time']}"
            else:
                t_display = h.get('timestamp_formatted', h.get('timestamp', '')[:19]).replace('\n', ' ')

            m = h.get('metrics', {})
            idx = len(hist) - i
            textbox.insert("end", f" Run #{idx}  —  {t_display}\n", "header")
            textbox.insert("end",
                f"   Acc:  {m.get('accuracy',0):.4f}    Prec: {m.get('precision',0):.4f}"
                f"    Rec:  {m.get('recall',0):.4f}    F1:   {m.get('f1',0):.4f}\n")
            textbox.insert("end",
                f"   TP: {m.get('tp',0):>4}  TN: {m.get('tn',0):>4}"
                f"  FP: {m.get('fp',0):>4}  FN: {m.get('fn',0):>4}"
                f"  Expert: {m.get('expert_ratio',0):.1%}\n")
            textbox.insert("end", "   " + "─" * 60 + "\n\n")

        textbox.configure(state="disabled")

    def view_logs(self):
        """Opens the debug_app.log using the OS default text editor."""
        if os.path.exists(self.log_file):
            try:
                if os.name == 'nt':
                    os.startfile(self.log_file)
                elif sys.platform == 'darwin':
                    import subprocess
                    subprocess.call(('open', self.log_file))
                else:
                    import subprocess
                    subprocess.call(('xdg-open', self.log_file))
            except Exception as e:
                messagebox.showerror("Error", f"Could not open logs: {e}")
        else:
            messagebox.showinfo("Logs", "No log file found.")

    # ==================== EXPORT DATA ====================
    def _export_combined_pdf_page(self, pdf):
        """
        Builds the PDF report:
          Page 1 — JSON filename header, pie donut + analysis summary side by side
          Pages 2+ — Appendix: one row per batch item (headline left, image right,
                      predicted/true/confidence/expert columns)
        """
        import textwrap
        import matplotlib.gridspec as gridspec

        BG     = "#1a1f2e"
        CARD   = "#242b3d"
        ACCENT = "#3b82f6"
        WHITE  = "#e2e8f0"
        GRAY   = "#8892a4"
        GREEN  = "#22c55e"
        RED    = "#ef4444"
        BLUE   = "#3498db"
        YELLOW = "#f39c12"

        json_fname = os.path.basename(getattr(self, 'json_path', '') or 'Unknown')
        tp = self.final_tp; tn = self.final_tn
        fp = self.final_fp; fn = self.final_fn
        total = tp + tn + fp + fn

        # ── PAGE 1: Summary ────────────────────────────────────────────────────
        fig1 = plt.figure(figsize=(8.27, 11.69), dpi=150)
        fig1.patch.set_facecolor(BG)

        outer = gridspec.GridSpec(
            2, 1, figure=fig1,
            height_ratios=[0.09, 0.88],
            hspace=0.03,
            left=0.07, right=0.93, top=0.97, bottom=0.03,
        )

        # Header row
        ax_hdr = fig1.add_subplot(outer[0])
        ax_hdr.set_facecolor(BG); ax_hdr.axis("off")
        ax_hdr.axhline(y=0.12, color=ACCENT, linewidth=2)
        ax_hdr.text(0.0, 0.95, "Multimodal Fake News Detector  —  Batch Analysis Report",
                    transform=ax_hdr.transAxes,
                    fontsize=14, fontweight="bold", color=WHITE, va="top",
                    fontfamily="DejaVu Sans")
        ax_hdr.text(0.0, 0.52, f"Source file: {json_fname}",
                    transform=ax_hdr.transAxes,
                    fontsize=8.5, color=ACCENT, va="top", style="italic",
                    fontfamily="DejaVu Sans")
        ax_hdr.text(1.0, 0.95,
                    datetime.now().strftime("%d %b %Y  %H:%M"),
                    transform=ax_hdr.transAxes,
                    fontsize=8.5, color=GRAY, va="top", ha="right",
                    fontfamily="DejaVu Sans")

        # Pie + Summary row
        body_gs = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer[1],
            width_ratios=[0.44, 0.56], wspace=0.06,
        )

        ax_pie = fig1.add_subplot(body_gs[0])
        ax_pie.set_facecolor(CARD); ax_pie.set_aspect("equal")

        labels_all = ["TP", "TN", "FP", "FN"]
        colors_all = [GREEN, BLUE, RED, YELLOW]
        vals_all   = [tp, tn, fp, fn]
        vals_f   = [v for v in vals_all if v > 0]
        lbls_f   = [l for v, l in zip(vals_all, labels_all) if v > 0]
        clrs_f   = [c for v, c in zip(vals_all, colors_all) if v > 0]

        if vals_f:
            _, _, autotexts = ax_pie.pie(
                vals_f, labels=lbls_f, colors=clrs_f, autopct="%1.1f%%",
                textprops={"color": WHITE, "fontsize": 9, "fontweight": "bold"},
                radius=1, wedgeprops=dict(width=0.46, edgecolor=BG, linewidth=1.5),
                startangle=90,
            )
            for at in autotexts: at.set_fontsize(8)
        else:
            ax_pie.pie([1], colors=[GRAY], radius=1, wedgeprops=dict(width=0.46, edgecolor=BG))

        ax_pie.text(0, 0, f"{total}\nItems", ha="center", va="center",
                    color=WHITE, fontsize=11, fontweight="bold")
        ax_pie.set_title("Prediction Distribution",
                         color=WHITE, fontsize=10, pad=10, fontweight="bold")
        for spine in ax_pie.spines.values(): spine.set_visible(False)

        ax_met = fig1.add_subplot(body_gs[1])
        ax_met.set_facecolor(CARD); ax_met.axis("off")
        for spine in ax_met.spines.values():
            spine.set_edgecolor(ACCENT); spine.set_linewidth(0.8); spine.set_visible(True)
        ax_met.text(0.05, 0.97, "Analysis Summary",
                    transform=ax_met.transAxes,
                    fontsize=11, fontweight="bold", color=ACCENT, va="top",
                    fontfamily="DejaVu Sans")
        ax_met.text(0.05, 0.88, self.report_text.strip(),
                    transform=ax_met.transAxes,
                    fontsize=8.5, color=WHITE, va="top",
                    linespacing=1.65, fontfamily="Courier New")

        fig1.text(0.5, 0.008,
                  "Fake News Detection System  •  Multimodal AI Analysis Report",
                  ha="center", fontsize=7, color=GRAY, style="italic",
                  fontfamily="DejaVu Sans")

        pdf.savefig(fig1, facecolor=BG, bbox_inches="tight")
        plt.close(fig1)

        # ── APPENDIX PAGES: 8 items per page ──────────────────────────────────
        results = getattr(self, 'batch_results_data', [])
        if not results:
            return

        ITEMS_PER_PAGE = 5
        pages = [results[i:i+ITEMS_PER_PAGE] for i in range(0, len(results), ITEMS_PER_PAGE)]
        total_pages = len(pages)

        def _square_crop(pil_img):
            """Centre-crops a PIL image to a square then resizes to 96x96."""
            w, h = pil_img.size
            m = min(w, h)
            left = (w - m) // 2; top = (h - m) // 2
            return pil_img.crop((left, top, left + m, top + m)).resize((96, 96), Image.Resampling.LANCZOS)

        for page_idx, page_items in enumerate(pages):
            padded = list(page_items) + [None] * (ITEMS_PER_PAGE - len(page_items))

            fig_ap = plt.figure(figsize=(8.27, 11.69), dpi=90)
            fig_ap.patch.set_facecolor(BG)

            ap_outer = gridspec.GridSpec(
                ITEMS_PER_PAGE + 1, 1, figure=fig_ap,
                height_ratios=[0.04] + [1.0] * ITEMS_PER_PAGE,
                hspace=0.06,
                left=0.04, right=0.96, top=0.98, bottom=0.02,
            )

            # Appendix page header
            ax_ap_hdr = fig_ap.add_subplot(ap_outer[0])
            ax_ap_hdr.set_facecolor(BG); ax_ap_hdr.axis("off")
            ax_ap_hdr.axhline(y=0.15, color=ACCENT, linewidth=1.0)
            ax_ap_hdr.text(0.0, 0.98,
                           f"Appendix — Detailed Predictions  (Page {page_idx+2} of {total_pages+1})",
                           transform=ax_ap_hdr.transAxes,
                           fontsize=10, fontweight="bold", color=WHITE, va="top",
                           fontfamily="DejaVu Sans")
            ax_ap_hdr.text(1.0, 0.98, f"Source: {json_fname}",
                           transform=ax_ap_hdr.transAxes,
                           fontsize=7.5, color=GRAY, va="top", ha="right",
                           style="italic", fontfamily="DejaVu Sans")

            for row_idx, item in enumerate(padded):
                # ── Empty / padding row ──────────────────────────────────────────
                if item is None:
                    ax_blank = fig_ap.add_subplot(ap_outer[row_idx + 1])
                    ax_blank.set_facecolor(BG); ax_blank.axis("off")
                    continue

                # ── Split row: text panel (left) | image panel (right) ───────────
                row_gs = gridspec.GridSpecFromSubplotSpec(
                    1, 2, subplot_spec=ap_outer[row_idx + 1],
                    width_ratios=[0.80, 0.20], wspace=0.02,
                )

                # ── Text panel ───────────────────────────────────────────────────
                ax_row = fig_ap.add_subplot(row_gs[0])
                ax_row.set_facecolor(CARD); ax_row.axis("off")
                for spine in ax_row.spines.values():
                    spine.set_edgecolor(ACCENT); spine.set_linewidth(0.6); spine.set_visible(True)

                pred_lbl = item.get("predicted_label", "?")
                true_lbl = item.get("true_label", "?")
                conf     = item.get("confidence", 0)
                expert   = item.get("expert_review", False)
                item_id  = item.get("id", "N/A")
                raw_text = item.get("text", "")[:220]
                headline = "\n".join(textwrap.wrap(raw_text, width=70))
                img_url  = item.get("image_url", "")

                correct    = pred_lbl == true_lbl
                status_sym = "✓" if correct else "✗"
                status_col = GREEN if correct else RED
                pred_color = RED if pred_lbl == "Fake" else GREEN
                expert_col = YELLOW if expert else GRAY
                expert_str = "Yes ⚑" if expert else "No"

                # ── Compact aligned block ────────────────────────────────────────
                # Constants
                PAD   = 0.018          # left margin (axes fraction)
                FS_ID = 8.5            # ID / status line
                FS    = 8.0            # body fields
                FS_LB = 7.5            # dim label size

                # Top bar: ID on left, status tick/cross on right
                ax_row.axhline(y=0.96, xmin=PAD, xmax=0.98,
                               color=ACCENT, linewidth=0.4, alpha=0.5)
                ax_row.text(PAD, 0.985, f"ID:  {item_id}",
                            transform=ax_row.transAxes,
                            fontsize=FS_ID, fontweight="bold",
                            color=WHITE, va="top",
                            fontfamily="DejaVu Sans")
                ax_row.text(0.985, 0.985, status_sym,
                            transform=ax_row.transAxes,
                            fontsize=FS_ID + 2, fontweight="bold",
                            color=status_col, va="top", ha="right")

                # News headline — italic, wraps naturally, indented under "News:"
                ax_row.text(PAD, 0.89, "News:",
                            transform=ax_row.transAxes,
                            fontsize=FS_LB, color=GRAY,
                            fontweight="bold", va="top",
                            fontfamily="DejaVu Sans")
                ax_row.text(PAD + 0.07, 0.89, headline,
                            transform=ax_row.transAxes,
                            fontsize=FS - 0.5, color="#cbd5e1", va="top",
                            linespacing=1.3, style="italic",
                            fontfamily="DejaVu Sans")

                # ── Metadata grid: 2-column, fixed label width ────────────────
                # Layout: each row = (y_position, label, value, value_color)
                fields = [
                    (0.41, "Predicted :", f"{pred_lbl:>8}",  pred_color),
                    (0.30, "True Label:", f"{true_lbl:>8}",  WHITE),
                    (0.19, "Confidence:", f"{conf:.1%}",     ACCENT),
                    (0.08, "Expert Rev:", expert_str,        expert_col),
                ]

                for y_f, lbl, val, vcol in fields:
                    # Separator micro-rule above each field
                    ax_row.axhline(y=y_f + 0.08, xmin=PAD, xmax=0.97,
                                   color=GRAY, linewidth=0.25, alpha=0.25)
                    ax_row.text(PAD, y_f, lbl,
                                transform=ax_row.transAxes,
                                fontsize=FS_LB, color=GRAY,
                                fontweight="bold", va="top",
                                fontfamily="Courier New")
                    ax_row.text(PAD + 0.17, y_f, val,
                                transform=ax_row.transAxes,
                                fontsize=FS, color=vcol,
                                fontweight="bold", va="top",
                                fontfamily="Courier New")

                # ── Image panel ──────────────────────────────────────────────────
                ax_im = fig_ap.add_subplot(row_gs[1])
                ax_im.set_facecolor(CARD)
                ax_im.set_xlim(0, 1); ax_im.set_ylim(0, 1)
                ax_im.set_xticks([]); ax_im.set_yticks([])
                for spine in ax_im.spines.values():
                    spine.set_edgecolor(ACCENT); spine.set_linewidth(0.5); spine.set_visible(True)

                MARGIN = 0.04
                local_img = item.get("img_path")
                if local_img and os.path.isfile(local_img):
                    try:
                        pil = _square_crop(Image.open(local_img).convert("RGB"))
                        ax_im.imshow(pil,
                                     extent=[MARGIN, 1-MARGIN, MARGIN, 1-MARGIN],
                                     aspect="equal", origin="upper",
                                     interpolation="bilinear")
                    except Exception:
                        ax_im.text(0.5, 0.5, "Error", ha="center", va="center",
                                   color=GRAY, fontsize=7)
                else:
                    ax_im.text(0.5, 0.5, "No\nImage", ha="center", va="center",
                               color=GRAY, fontsize=7, style="italic", linespacing=1.4)

            pdf.savefig(fig_ap, facecolor=BG, bbox_inches="tight")
            plt.close(fig_ap)

    def _export_metrics_png(self, out_folder):
        """
        Saves 4 raw PNG graphs into out_folder (one file per metric panel):
          1_pie.png           - Prediction distribution donut
          2_confusion.png     - Confusion matrix heatmap
          3_metrics_bar.png   - Acc / Prec / Rec / F1 horizontal bars
          4_kpi_scorecard.png - KPI count tiles
        """
        import matplotlib.gridspec as gridspec
        import numpy as np

        BG     = "#1a1f2e"
        CARD   = "#242b3d"
        ACCENT = "#3b82f6"
        WHITE  = "#e2e8f0"
        GRAY   = "#8892a4"
        GREEN  = "#22c55e"
        RED    = "#ef4444"
        BLUE   = "#3498db"
        YELLOW = "#f39c12"

        tp = self.final_tp;  tn = self.final_tn
        fp = self.final_fp;  fn = self.final_fn
        total = tp + tn + fp + fn
        acc  = (tp + tn) / total if total > 0 else 0
        prec = tp / (tp + fp)   if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn)   if (tp + fn) > 0 else 0
        f1   = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        expert = getattr(self, "final_expert", 0)

        def _savefig(fig, name):
            fig.savefig(os.path.join(out_folder, name),
                        facecolor=BG, bbox_inches="tight", dpi=150)
            plt.close(fig)

        # ── 1. Donut Pie ─────────────────────────────────────────────────────
        fig1, ax = plt.subplots(figsize=(6, 6), dpi=150)
        fig1.patch.set_facecolor(BG); ax.set_facecolor(CARD); ax.set_aspect("equal")
        labels_all = ["TP", "TN", "FP", "FN"]
        colors_all = [GREEN, BLUE, RED, YELLOW]
        vals_all   = [tp, tn, fp, fn]
        fv = [v for v in vals_all if v > 0]
        fl = [l for v, l in zip(vals_all, labels_all) if v > 0]
        fc = [c for v, c in zip(vals_all, colors_all) if v > 0]
        if fv:
            _, _, ats = ax.pie(fv, labels=fl, colors=fc, autopct="%1.1f%%",
                textprops={"color": WHITE, "fontsize": 10, "fontweight": "bold"},
                radius=1, wedgeprops=dict(width=0.48, edgecolor=BG, linewidth=2),
                startangle=90)
            for at in ats: at.set_fontsize(9)
        else:
            ax.pie([1], colors=[GRAY], radius=1, wedgeprops=dict(width=0.48, edgecolor=BG))
        ax.text(0, 0, f"{total}\nItems", ha="center", va="center",
                color=WHITE, fontsize=12, fontweight="bold")
        ax.set_title("Prediction Distribution", color=WHITE, fontsize=12,
                     pad=12, fontweight="bold")
        _savefig(fig1, "1_pie.png")

        # ── 2. Confusion Matrix ───────────────────────────────────────────────
        fig2, ax = plt.subplots(figsize=(5, 5), dpi=150)
        fig2.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, int(v), ha="center", va="center",
                    fontsize=16, fontweight="bold",
                    color=WHITE if cm[r, c] > cm.max() * 0.4 else GRAY)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Real", "Pred Fake"], color=WHITE, fontsize=10)
        ax.set_yticklabels(["Act Real", "Act Fake"], color=WHITE, fontsize=10,
                           rotation=90, va="center")
        ax.tick_params(colors=WHITE, length=0)
        for sp in ax.spines.values(): sp.set_edgecolor(ACCENT); sp.set_linewidth(1)
        ax.set_title("Confusion Matrix", color=WHITE, fontsize=12,
                     pad=10, fontweight="bold")
        _savefig(fig2, "2_confusion.png")

        # ── 3. Metrics Bar Chart ──────────────────────────────────────────────
        fig3, ax = plt.subplots(figsize=(7, 4), dpi=150)
        fig3.patch.set_facecolor(BG); ax.set_facecolor(CARD)
        names  = ["Accuracy", "Precision", "Recall", "F1 Score"]
        values = [acc, prec, rec, f1]
        bcolors = [BLUE, GREEN, YELLOW, ACCENT]
        bars = ax.barh(names, values, color=bcolors, height=0.5, edgecolor=BG)
        for bar, val in zip(bars, values):
            ax.text(min(val + 0.02, 0.96), bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=10,
                    fontweight="bold", color=WHITE)
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=GRAY, fontsize=9)
        ax.tick_params(colors=WHITE, length=0)
        ax.yaxis.set_tick_params(labelcolor=WHITE, labelsize=11)
        for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]: ax.spines[sp].set_edgecolor(ACCENT)
        ax.axvline(0.5, color=GRAY, linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title("Classification Metrics", color=WHITE, fontsize=12,
                     pad=10, fontweight="bold")
        _savefig(fig3, "3_metrics_bar.png")

        # ── 4. KPI Scorecard ──────────────────────────────────────────────────
        fig4, ax = plt.subplots(figsize=(10, 3), dpi=150)
        fig4.patch.set_facecolor(BG); ax.set_facecolor(BG); ax.axis("off")
        kpis = [
            ("True Positives",  tp,    GREEN),
            ("True Negatives",  tn,    BLUE),
            ("False Positives", fp,    RED),
            ("False Negatives", fn,    YELLOW),
            ("Total Items",     total, WHITE),
            ("Expert Reviews",  expert, ACCENT),
        ]
        tile_w, tile_h, pad_x = 0.148, 0.72, 0.012
        for i, (lbl, val, col) in enumerate(kpis):
            x = 0.01 + i * (tile_w + pad_x)
            ax.add_patch(plt.Rectangle((x, 0.08), tile_w, tile_h,
                         transform=ax.transAxes, facecolor=CARD,
                         edgecolor=col, linewidth=1.5, clip_on=False))
            ax.text(x + tile_w / 2, 0.60, str(val),
                    transform=ax.transAxes, fontsize=20,
                    fontweight="bold", color=col, ha="center", va="center")
            ax.text(x + tile_w / 2, 0.20, lbl,
                    transform=ax.transAxes, fontsize=8,
                    color=GRAY, ha="center", va="center")
        ax.set_title("KPI Scorecard", color=WHITE, fontsize=12,
                     pad=8, fontweight="bold")
        _savefig(fig4, "4_kpi_scorecard.png")

    def export_results(self):
        """
        Exports batch analysis results and graphs as PDF, JSON, or PNG.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        file_path = filedialog.asksaveasfilename(
            title="Export Results",
            initialfile=f"result_{timestamp}",
            defaultextension=".pdf",
            filetypes=[
                ("PDF Report", "*.pdf"),
                ("JSON Data", "*.json"),
                ("PNG Graph", "*.png"),
            ],
        )
        if not file_path:
            return

        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                with PdfPages(file_path) as pdf:
                    self._export_combined_pdf_page(pdf)

                messagebox.showinfo("Export Success", f"PDF exported to:\n{file_path}")
                logging.info(f"PDF exported to {file_path}")

            elif ext == ".json":
                # Build export payload: keep text + image_url, drop internal img_path
                export_data = [
                    {k: v for k, v in row.items() if k != "img_path"}
                    for row in self.batch_results_data
                ]
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(export_data, f, indent=4, ensure_ascii=False)

                messagebox.showinfo("Export Success", f"JSON exported to:\n{file_path}")
                logging.info(f"JSON exported to {file_path}")

            elif ext == ".png":
                # Folder name = user-chosen filename stem + "_graphs"
                # e.g. saving "report_20260430.png" → folder "report_20260430_graphs/"
                chosen_stem = os.path.splitext(os.path.basename(file_path))[0]
                base_dir    = os.path.dirname(file_path)
                out_folder  = os.path.join(base_dir, chosen_stem + "_graphs")
                os.makedirs(out_folder, exist_ok=True)

                self._export_metrics_png(out_folder)

                messagebox.showinfo(
                    "Export Success",
                    f"4 PNG graphs saved to folder:\n{out_folder}"
                )
                logging.info(f"PNG graphs exported to folder: {out_folder}")

            else:
                messagebox.showerror("Export Error", "Unsupported export format selected.")

        except Exception as e:
            logging.error(f"Failed to export: {e}")
            messagebox.showerror("Export Error", f"Failed to export: {str(e)}")

if __name__ == "__main__":
    app = FakeNewsApp()
    app.mainloop()