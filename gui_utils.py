"""
GUI utility functions for the Information Retrieval system.
"""

import tkinter as tk
from tkinter import ttk
import threading
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import io
from typing import Callable, Optional


class ThreadedTask:
    """Helper class for running tasks in background threads."""
    
    def __init__(self, target_func: Callable, callback: Optional[Callable] = None, root=None):
        """
        Initialize threaded task.
        
        Args:
            target_func: Function to run in background
            callback: Function to call when task completes
            root: Tkinter root window for scheduling callbacks on main thread
        """
        self.target_func = target_func
        self.callback = callback
        self.root = root
        self.thread = None
    
    def start(self):
        """Start the task in a background thread."""
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def _run(self):
        """Run the task and call callback."""
        try:
            result = self.target_func()
            if self.callback:
                # Schedule callback on main thread if root is provided
                if self.root:
                    self.root.after(0, lambda r=result: self.callback(r))
                else:
                    self.callback(result)
        except Exception as e:
            if self.callback:
                error_result = {'error': str(e)}
                if self.root:
                    self.root.after(0, lambda r=error_result: self.callback(r))
                else:
                    self.callback(error_result)


def create_progress_window(parent, title: str = "Processing...", determinate: bool = False):
    """
    Create a progress window with a progress bar.
    
    Args:
        parent: Parent window
        title: Window title
        determinate: If True, use determinate mode (0-100), else indeterminate
        
    Returns:
        Tuple of (window, progress_bar, status_label)
    """
    progress_window = tk.Toplevel(parent)
    progress_window.title(title)
    progress_window.geometry("500x140")
    progress_window.transient(parent)
    progress_window.grab_set()
    
    # Center the window
    progress_window.update_idletasks()
    x = (progress_window.winfo_screenwidth() // 2) - (500 // 2)
    y = (progress_window.winfo_screenheight() // 2) - (140 // 2)
    progress_window.geometry(f"500x140+{x}+{y}")
    
    frame = ttk.Frame(progress_window, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)
    
    status_label = ttk.Label(frame, text="Initializing...", font=("Arial", 10))
    status_label.pack(pady=5)
    
    if determinate:
        progress_bar = ttk.Progressbar(frame, mode='determinate', maximum=100, length=400)
        progress_bar.pack(fill=tk.X, pady=10)
        progress_bar['value'] = 0
    else:
        progress_bar = ttk.Progressbar(frame, mode='indeterminate', length=400)
        progress_bar.pack(fill=tk.X, pady=10)
        progress_bar.start()
    
    return progress_window, progress_bar, status_label


def update_status_label(label: ttk.Label, text: str):
    """Update status label text (thread-safe)."""
    label.after(0, lambda: label.config(text=text))


def create_matplotlib_canvas(parent, figsize=(6, 4)):
    """
    Create a matplotlib canvas embedded in Tkinter.
    
    Args:
        parent: Parent widget
        figsize: Figure size tuple
        
    Returns:
        Tuple of (figure, canvas)
    """
    fig = Figure(figsize=figsize, dpi=100)
    canvas = FigureCanvasTkAgg(fig, parent)
    canvas.draw()
    return fig, canvas


def clear_canvas(canvas):
    """Clear a matplotlib canvas."""
    canvas.figure.clear()
    canvas.draw()


def display_image_in_label(label: tk.Label, image_path: str, max_size=(400, 300)):
    """
    Display an image in a Tkinter label.
    
    Args:
        label: Label widget to display image
        image_path: Path to image file
        max_size: Maximum size for image
    """
    try:
        img = Image.open(image_path)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo  # Keep a reference
    except Exception as e:
        label.config(text=f"Error loading image: {e}")


def display_pil_image_in_label(label: tk.Label, pil_image: Image.Image, max_size=(400, 300)):
    """
    Display a PIL Image in a Tkinter label.
    
    Args:
        label: Label widget to display image
        pil_image: PIL Image object
        max_size: Maximum size for image
    """
    try:
        img = pil_image.copy()
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        label.config(image=photo)
        label.image = photo  # Keep a reference
    except Exception as e:
        label.config(text=f"Error displaying image: {e}")


def format_document_preview(text: str, max_length: int = 200) -> str:
    """
    Format document text for preview.
    
    Args:
        text: Document text
        max_length: Maximum preview length
        
    Returns:
        Formatted preview string
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_score(score: float, decimals: int = 4) -> str:
    """
    Format a score for display.
    
    Args:
        score: Score value
        decimals: Number of decimal places
        
    Returns:
        Formatted score string
    """
    return f"{score:.{decimals}f}"


def create_tooltip(widget, text: str):
    """
    Create a tooltip for a widget.
    
    Args:
        widget: Widget to attach tooltip to
        text: Tooltip text
    """
    def on_enter(event):
        tooltip = tk.Toplevel()
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        label = tk.Label(tooltip, text=text, background="yellow", 
                        relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
        label.pack()
        widget.tooltip = tooltip
    
    def on_leave(event):
        if hasattr(widget, 'tooltip'):
            widget.tooltip.destroy()
            del widget.tooltip
    
    widget.bind('<Enter>', on_enter)
    widget.bind('<Leave>', on_leave)


def create_scrollable_frame(parent):
    """
    Create a scrollable frame.
    
    Args:
        parent: Parent widget
        
    Returns:
        Tuple of (canvas, scrollable_frame)
    """
    canvas = tk.Canvas(parent)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)
    
    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )
    
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)
    
    return canvas, scrollable_frame, scrollbar


def safe_tk_call(func: Callable, *args, **kwargs):
    """
    Safely call a Tkinter function from a non-main thread.
    
    Args:
        func: Function to call
        *args: Positional arguments
        **kwargs: Keyword arguments
    """
    def call():
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(f"Error in safe_tk_call: {e}")
    
    if hasattr(func, 'after'):
        func.after(0, call)
    else:
        call()

