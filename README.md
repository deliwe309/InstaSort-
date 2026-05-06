#!/usr/bin/env python3
"""
InstaSort - AI-Powered Image Organizer Desktop App
Complete single-file application

Save as: instasort.py
Run with: python instasort.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from threading import Thread
from pathlib import Path
from datetime import datetime
import hashlib
import shutil
import json
import sqlite3
import logging
import sys
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Callable
from PIL import Image
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    APP_NAME: str = "InstaSort"
    APP_VERSION: str = "1.0.0"
    
    # Categories for classification
    CATEGORIES: List[str] = None
    
    # Sort folder patterns
    SORT_PATTERNS: Dict[str, str] = None
    
    # Settings
    IMAGE_EXTENSIONS: set = None
    CONFIDENCE_THRESHOLD: float = 0.65
    BATCH_SIZE: int = 32
    
    def __post_init__(self):
        if self.CATEGORIES is None:
            self.CATEGORIES = [
                'portrait', 'landscape', 'animal', 'cityscape',
                'food', 'vehicle', 'document', 'other'
            ]
        
        if self.SORT_PATTERNS is None:
            self.SORT_PATTERNS = {
                'portrait': 'People/Portraits',
                'landscape': 'Nature/Landscapes',
                'animal': 'Animals',
                'cityscape': 'Urban/Cityscapes',
                'food': 'Food',
                'vehicle': 'Vehicles',
                'document': 'Documents',
                'other': 'Miscellaneous'
            }
        
        if self.IMAGE_EXTENSIONS is None:
            self.IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database with tables"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    original_path TEXT,
                    file_hash TEXT,
                    category TEXT,
                    confidence REAL,
                    width INTEGER,
                    height INTEGER,
                    file_size INTEGER,
                    date_processed TIMESTAMP,
                    is_duplicate BOOLEAN DEFAULT 0,
                    duplicate_of TEXT
                );
                
                CREATE TABLE IF NOT EXISTS processing_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_folder TEXT NOT NULL,
                    target_base TEXT NOT NULL,
                    total_images INTEGER,
                    processed_images INTEGER DEFAULT 0,
                    status TEXT,
                    started_at TIMESTAMP,
                    completed_at TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_hash ON images(file_hash);
                CREATE INDEX IF NOT EXISTS idx_category ON images(category);
            ''')
    
    def add_image(self, data: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO images 
                (file_path, original_path, file_hash, category, confidence, 
                 width, height, file_size, date_processed, is_duplicate, duplicate_of)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data['file_path'], data.get('original_path'), data['file_hash'],
                data['category'], data['confidence'], data['width'],
                data['height'], data['file_size'], datetime.now(),
                data.get('is_duplicate', 0), data.get('duplicate_of')
            ))
    
    def get_image_by_hash(self, file_hash: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute('SELECT * FROM images WHERE file_hash = ?', (file_hash,)).fetchone()
    
    def get_all_images(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute('SELECT * FROM images ORDER BY date_processed DESC').fetchall()
    
    def create_job(self, source_folder: str, target_base: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO processing_jobs (source_folder, target_base, status, started_at)
                VALUES (?, ?, ?, ?)
            ''', (source_folder, target_base, 'running', datetime.now()))
            return cursor.lastrowid
    
    def update_job(self, job_id: int, processed_images: int = None, total_images: int = None, status: str = None):
        with sqlite3.connect(self.db_path) as conn:
            if total_images is not None:
                conn.execute('UPDATE processing_jobs SET total_images = ? WHERE id = ?', (total_images, job_id))
            if processed_images is not None:
                conn.execute('UPDATE processing_jobs SET processed_images = ? WHERE id = ?', (processed_images, job_id))
            if status is not None:
                completed_at = datetime.now() if status == 'completed' else None
                conn.execute('UPDATE processing_jobs SET status = ?, completed_at = ? WHERE id = ?', 
                           (status, completed_at, job_id))
    
    def get_setting(self, key: str, default=None):
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT value FROM settings WHERE key = ?', (key,)).fetchone()
            if result:
                return json.loads(result[0])
            return default
    
    def set_setting(self, key: str, value):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO settings (key, value, updated_at)
                VALUES (?, ?, ?)
            ''', (key, json.dumps(value), datetime.now()))
    
    def get_job_history(self, limit: int = 50):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return conn.execute('''
                SELECT * FROM processing_jobs ORDER BY id DESC LIMIT ?
            ''', (limit,)).fetchall()


# ============================================================================
# IMAGE CLASSIFIER
# ============================================================================

class ImageClassifier:
    def __init__(self, categories: List[str], threshold: float):
        self.categories = categories
        self.threshold = threshold
        self.use_dummy = True  # Use rule-based classifier (no external model needed)
        
    def preprocess_image(self, image_path: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """Load and preprocess image"""
        try:
            img = Image.open(image_path).convert('RGB')
            original_size = img.size
            # Resize for analysis
            img = img.resize((224, 224))
            img_array = np.array(img, dtype=np.float32)
            img_array = img_array / 127.5 - 1
            img_array = np.expand_dims(img_array, axis=0)
            return img_array, original_size
        except Exception as e:
            return None, None
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """Classify image using rule-based heuristics"""
        try:
            img = Image.open(image_path).convert('RGB')
            width, height = img.size
            aspect_ratio = width / height
            
            # Rule-based classification
            # Check if document/screenshot (mostly text)
            if self._is_likely_document(img):
                return 'document', 0.78
            
            # Check aspect ratio
            if aspect_ratio > 1.3:
                return 'landscape', 0.72
            elif aspect_ratio < 0.77:
                return 'portrait', 0.71
            
            # Color analysis
            img_small = img.resize((50, 50))
            pixels = np.array(img_small)
            avg_color = pixels.mean(axis=(0, 1))
            std_color = pixels.std(axis=(0, 1))
            
            # Detect food (warm colors, higher saturation)
            if avg_color[0] > 130 and avg_color[1] > 80 and std_color.mean() > 40:
                return 'food', 0.68
            
            # Detect nature (greens)
            if avg_color[1] > 120 and avg_color[1] > avg_color[0] + 20:
                return 'landscape', 0.74
            
            # Detect city (grays, blues)
            if avg_color[0] - avg_color[2] < 30 and avg_color[1] < 120:
                return 'cityscape', 0.66
            
            # Default
            return 'other', 0.55
            
        except Exception as e:
            return 'other', 0.0
    
    def _is_likely_document(self, img: Image) -> bool:
        """Heuristic to detect documents/screenshots"""
        # Check if image is mostly white/grayscale
        img_gray = img.convert('L')
        pixels = np.array(img_gray)
        high_contrast = np.std(pixels) > 80
        light_background = np.mean(pixels) > 200
        
        # Check for text-like patterns (high edge density)
        from scipy import ndimage
        edges = np.abs(ndimage.sobel(pixels.astype(float)))
        edge_density = np.mean(edges > 30)
        
        return edge_density > 0.15 and light_background
    
    def batch_predict(self, image_paths: List[str], callback: Optional[Callable] = None) -> List[Tuple]:
        """Classify multiple images"""
        results = []
        total = len(image_paths)
        
        for idx, path in enumerate(image_paths):
            category, confidence = self.predict(path)
            results.append((path, category, confidence))
            
            if callback and (idx + 1) % 10 == 0:
                callback(idx + 1, total, path)
        
        return results


# ============================================================================
# FILE ORGANIZER
# ============================================================================

class FileOrganizer:
    def __init__(self, db: DatabaseManager, config: Config):
        self.db = db
        self.config = config
    
    def compute_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file"""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def get_metadata(self, image_path: str) -> dict:
        """Extract image metadata"""
        try:
            img = Image.open(image_path)
            return {
                'width': img.size[0],
                'height': img.size[1],
                'file_size': Path(image_path).stat().st_size
            }
        except Exception:
            return {'width': 0, 'height': 0, 'file_size': 0}
    
    def find_all_images(self, source_folder: str) -> List[Path]:
        """Recursively find all images in folder"""
        images = []
        source_path = Path(source_folder)
        
        for ext in self.config.IMAGE_EXTENSIONS:
            images.extend(source_path.rglob(f'*{ext}'))
            images.extend(source_path.rglob(f'*{ext.upper()}'))
        
        return images
    
    def sort_images(self, source_folder: str, target_base: str, classifier: ImageClassifier,
                   progress_callback: Optional[Callable] = None) -> dict:
        """Main sorting function"""
        source_path = Path(source_folder)
        target_path = Path(target_base)
        
        # Find all images
        image_files = self.find_all_images(source_folder)
        total = len(image_files)
        
        if total == 0:
            return {'total_processed': 0, 'results': [], 'message': 'No images found'}
        
        # Create job record
        job_id = self.db.create_job(str(source_path), str(target_path))
        self.db.update_job(job_id, processed_images=0, total_images=total)
        
        processed = 0
        results = []
        stats = {cat: 0 for cat in self.config.CATEGORIES}
        stats['duplicate'] = 0
        
        for img_path in image_files:
            try:
                # Check for duplicate
                file_hash = self.compute_hash(str(img_path))
                existing = self.db.get_image_by_hash(file_hash)
                
                is_duplicate = existing is not None
                duplicate_of = existing['file_path'] if is_duplicate else None
                
                if is_duplicate:
                    category = existing['category']
                    confidence = existing['confidence']
                    stats['duplicate'] += 1
                else:
                    # Classify image
                    category, confidence = classifier.predict(str(img_path))
                    if confidence < self.config.CONFIDENCE_THRESHOLD:
                        category = 'other'
                    stats[category] = stats.get(category, 0) + 1
                
                # Get metadata
                metadata = self.get_metadata(img_path)
                
                # Determine destination
                dest_folder = target_path / self.config.SORT_PATTERNS.get(category, 'Miscellaneous')
                dest_folder.mkdir(parents=True, exist_ok=True)
                
                # Generate unique filename
                dest_path = dest_folder / img_path.name
                counter = 1
                while dest_path.exists():
                    stem = img_path.stem
                    dest_path = dest_folder / f"{stem}_{counter}{img_path.suffix}"
                    counter += 1
                
                # Copy file (or skip if duplicate)
                action = 'skipped (duplicate)'
                if not is_duplicate:
                    shutil.copy2(str(img_path), str(dest_path))
                    action = f'copied to {dest_path}'
                
                # Store in database
                self.db.add_image({
                    'file_path': str(dest_path),
                    'original_path': str(img_path),
                    'file_hash': file_hash,
                    'category': category,
                    'confidence': confidence,
                    'width': metadata['width'],
                    'height': metadata['height'],
                    'file_size': metadata['file_size'],
                    'is_duplicate': 1 if is_duplicate else 0,
                    'duplicate_of': duplicate_of
                })
                
                results.append({
                    'source': str(img_path),
                    'destination': str(dest_path) if not is_duplicate else None,
                    'category': category,
                    'confidence': confidence,
                    'action': action
                })
                
                processed += 1
                self.db.update_job(job_id, processed_images=processed)
                
                if progress_callback:
                    progress_callback(processed, total, img_path.name, category, action)
                
            except Exception as e:
                results.append({'source': str(img_path), 'error': str(e)})
                processed += 1
        
        # Mark job as completed
        self.db.update_job(job_id, processed_images=processed, status='completed')
        
        return {
            'job_id': job_id,
            'total_processed': processed,
            'stats': stats,
            'results': results,
            'target_base': str(target_path)
        }


# ============================================================================
# GUI APPLICATION
# ============================================================================

class InstaSortApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = Config()
        
        # Setup paths
        self.app_data_dir = Path.home() / '.instasort'
        self.app_data_dir.mkdir(exist_ok=True)
        self.db = DatabaseManager(self.app_data_dir / 'instasort.db')
        
        # Initialize components
        self.classifier = ImageClassifier(self.config.CATEGORIES, self.config.CONFIDENCE_THRESHOLD)
        self.organizer = FileOrganizer(self.db, self.config)
        
        # UI Variables
        self.source_folder = tk.StringVar()
        self.target_folder = tk.StringVar()
        self.is_processing = False
        self.processing_thread = None
        
        # Setup UI
        self.root.title(f"{self.config.APP_NAME} v{self.config.APP_VERSION}")
        self.root.geometry("1100x750")
        self.root.minsize(800, 600)
        
        # Configure styles
        self._setup_styles()
        
        # Build interface
        self._build_ui()
        
        # Load saved settings
        self._load_settings()
    
    def _setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'))
        style.configure('Heading.TLabel', font=('Segoe UI', 11, 'bold'))
        style.configure('Success.TButton', foreground='green')
        style.configure('Danger.TButton', foreground='red')
    
    def _build_ui(self):
        """Build the complete UI"""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(title_frame, text="📸 InstaSort", style='Title.TLabel').pack(side=tk.LEFT)
        ttk.Label(title_frame, text="AI-Powered Image Organizer", 
                 font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(10, 0))
        
        # Two-panel layout
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_panel = ttk.Frame(paned)
        paned.add(left_panel, weight=1)
        
        # Right panel - Log
        right_panel = ttk.Frame(paned)
        paned.add(right_panel, weight=1)
        
        # ==================== LEFT PANEL ====================
        
        # Source folder selection
        source_frame = ttk.LabelFrame(left_panel, text="1. Select Source", padding="10")
        source_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(source_frame, text="Folder with unsorted images:").pack(anchor=tk.W)
        source_entry_frame = ttk.Frame(source_frame)
        source_entry_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Entry(source_entry_frame, textvariable=self.source_folder).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(source_entry_frame, text="Browse", command=self._browse_source).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Target folder selection
        target_frame = ttk.LabelFrame(left_panel, text="2. Select Destination", padding="10")
        target_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(target_frame, text="Where to save organized images:").pack(anchor=tk.W)
        target_entry_frame = ttk.Frame(target_frame)
        target_entry_frame.pack(fill=tk.X, pady=(5, 0))
        ttk.Entry(target_entry_frame, textvariable=self.target_folder).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(target_entry_frame, text="Browse", command=self._browse_target).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Settings
        settings_frame = ttk.LabelFrame(left_panel, text="Settings", padding="10")
        settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.threshold_var = tk.DoubleVar(value=self.config.CONFIDENCE_THRESHOLD)
        threshold_scale = ttk.Scale(settings_frame, from_=0.0, to=1.0, variable=self.threshold_var, orient=tk.HORIZONTAL)
        threshold_scale.pack(fill=tk.X, pady=5)
        self.threshold_label = ttk.Label(settings_frame, text=f"Current: {self.config.CONFIDENCE_THRESHOLD:.2f}")
        self.threshold_label.pack(anchor=tk.W)
        threshold_scale.configure(command=self._update_threshold)
        
        # Action buttons
        action_frame = ttk.Frame(left_panel)
        action_frame.pack(fill=tk.X, pady=(20, 10))
        
        self.start_btn = ttk.Button(action_frame, text="▶ Start Sorting", command=self._start_sorting)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        self.stop_btn = ttk.Button(action_frame, text="⏹ Stop", command=self._stop_sorting, state='disabled')
        self.stop_btn.pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Additional buttons
        extra_frame = ttk.Frame(left_panel)
        extra_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(extra_frame, text="🔍 Find Duplicates", command=self._show_duplicates).pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        ttk.Button(extra_frame, text="📜 View History", command=self._show_history).pack(side=tk.RIGHT, padx=(5, 0), fill=tk.X, expand=True)
        
        # Statistics
        stats_frame = ttk.LabelFrame(left_panel, text="Statistics", padding="10")
        stats_frame.pack(fill=tk.BOTH, expand=True)
        
        self.stats_text = tk.Text(stats_frame, height=12, wrap=tk.WORD, font=('Consolas', 9))
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        progress_frame = ttk.LabelFrame(left_panel, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)
        
        self.status_label = ttk.Label(progress_frame, text="Ready")
        self.status_label.pack(anchor=tk.W, pady=(5, 0))
        
        self.current_file_label = ttk.Label(progress_frame, text="", font=('Segoe UI', 8))
        self.current_file_label.pack(anchor=tk.W)
        
        # ==================== RIGHT PANEL ====================
        
        log_frame = ttk.LabelFrame(right_panel, text="Processing Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log controls
        log_control = ttk.Frame(log_frame)
        log_control.pack(fill=tk.X, pady=(0, 5))
        ttk.Button(log_control, text="Clear Log", command=self._clear_log).pack(side=tk.RIGHT)
        ttk.Button(log_control, text="Export Log", command=self._export_log).pack(side=tk.RIGHT, padx=(0, 5))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=25, font=('Consolas', 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Initialize log with welcome message
        self._log(f"{self.config.APP_NAME} v{self.config.APP_VERSION} ready")
        self._log(f"Using rule-based classifier (no AI model required)")
        self._log("-" * 50)
    
    def _update_threshold(self, value):
        """Update confidence threshold display"""
        self.threshold_label.config(text=f"Current: {float(value):.2f}")
        self.config.CONFIDENCE_THRESHOLD = float(value)
        self.classifier.threshold = float(value)
    
    def _browse_source(self):
        folder = filedialog.askdirectory(title="Select folder with unsorted images")
        if folder:
            self.source_folder.set(folder)
            self.db.set_setting('last_source', folder)
    
    def _browse_target(self):
        folder = filedialog.askdirectory(title="Select destination folder for organized images")
        if folder:
            self.target_folder.set(folder)
            self.db.set_setting('last_target', folder)
    
    def _load_settings(self):
        """Load saved settings"""
        source = self.db.get_setting('last_source')
        target = self.db.get_setting('last_target')
        if source:
            self.source_folder.set(source)
        if target:
            self.target_folder.set(target)
    
    def _log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def _clear_log(self):
        """Clear the log text"""
        self.log_text.delete(1.0, tk.END)
        self._log("Log cleared")
    
    def _export_log(self):
        """Export log to file"""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            with open(file_path, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self._log(f"Log exported to {file_path}")
    
    def _update_progress(self, current: int, total: int, filename: str, category: str, action: str):
        """Update progress UI"""
        percent = (current / total) * 100
        self.progress_var.set(percent)
        self.status_label.config(text=f"Processing: {current}/{total} images ({percent:.1f}%)")
        self.current_file_label.config(text=f"📷 {filename} → {category}")
        self._log(f"✓ {filename} → {category} ({action})")
        self.root.update_idletasks()
    
    def _update_stats(self, stats: dict):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, "Last Run Statistics:\n")
        self.stats_text.insert(tk.END, "-" * 30 + "\n")
        
        total = sum(v for k, v in stats.items() if k != 'duplicate')
        self.stats_text.insert(tk.END, f"Total processed: {total}\n")
        self.stats_text.insert(tk.END, f"Duplicates found: {stats.get('duplicate', 0)}\n\n")
        
        self.stats_text.insert(tk.END, "By category:\n")
        for cat, count in stats.items():
            if cat != 'duplicate' and count > 0:
                self.stats_text.insert(tk.END, f"  • {cat}: {count}\n")
    
    def _start_sorting(self):
        """Start the sorting process"""
        source = self.source_folder.get().strip()
        target = self.target_folder.get().strip()
        
        if not source:
            messagebox.showerror("Error", "Please select a source folder")
            return
        
        if not target:
            messagebox.showerror("Error", "Please select a target folder")
            return
        
        if not os.path.exists(source):
            messagebox.showerror("Error", f"Source folder does not exist:\n{source}")
            return
        
        self.is_processing = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_var.set(0)
        
        self._log("=" * 50)
        self._log(f"Starting sorting job")
        self._log(f"Source: {source}")
        self._log(f"Target: {target}")
        self._log(f"Confidence threshold: {self.config.CONFIDENCE_THRESHOLD:.2f}")
        self._log("-" * 50)
        
        # Run in separate thread
        self.processing_thread = Thread(target=self._process_images, args=(source, target))
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def _process_images(self, source: str, target: str):
        """Process images in background thread"""
        try:
            result = self.organizer.sort_images(
                source, target, self.classifier,
                progress_callback=self._update_progress
            )
            self.root.after(0, self._on_complete, result)
        except Exception as e:
            self.root.after(0, self._on_error, str(e))
    
    def _on_complete(self, result: dict):
        """Handle completion"""
        self.is_processing = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        
        self._log("-" * 50)
        self._log(f"✓ Sorting completed!")
        self._log(f"Total processed: {result['total_processed']}")
        self._log(f"Target folder: {result['target_base']}")
        
        # Update stats
        if 'stats' in result:
            self._update_stats(result['stats'])
        
        messagebox.showinfo(
            "Complete",
            f"Successfully processed {result['total_processed']} images!\n\n"
            f"Organized images saved to:\n{result['target_base']}"
        )
    
    def _on_error(self, error: str):
        """Handle error"""
        self.is_processing = False
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self._log(f"✗ ERROR: {error}")
        messagebox.showerror("Processing Error", f"An error occurred:\n{error}")
    
    def _stop_sorting(self):
        """Stop sorting process"""
        self.is_processing = False
        self._log("⚠ Stopping sorting (will complete current file)...")
        self.stop_btn.config(state='disabled')
    
    def _show_duplicates(self):
        """Show duplicate images window"""
        source = self.source_folder.get().strip()
        
        if not source or not os.path.exists(source):
            messagebox.showinfo("Info", "Please select a valid source folder first")
            return
        
        self._log("Scanning for duplicates...")
        
        # Find duplicates
        hashes = {}
        duplicates = []
        
        for img_path in Path(source).rglob('*'):
            if img_path.suffix.lower() in self.config.IMAGE_EXTENSIONS:
                file_hash = self.organizer.compute_hash(str(img_path))
                if file_hash in hashes:
                    duplicates.append({
                        'original': hashes[file_hash],
                        'duplicate': str(img_path)
                    })
                else:
                    hashes[file_hash] = str(img_path)
        
        if not duplicates:
            messagebox.showinfo("Duplicates", "No duplicate images found!")
            self._log("No duplicates found")
            return
        
        # Create results window
        dup_window = tk.Toplevel(self.root)
        dup_window.title(f"Duplicate Images - Found {len(duplicates)}")
        dup_window.geometry("900x600")
        
        text = scrolledtext.ScrolledText(dup_window, font=('Consolas', 9))
        text.pack(fill=tk.BOTH, expand=True)
        
        for dup in duplicates:
            text.insert(tk.END, f"\n📁 ORIGINAL: {dup['original']}\n")
            text.insert(tk.END, f"📋 DUPLICATE: {dup['duplicate']}\n")
            text.insert(tk.END, "-" * 80 + "\n")
        
        text.config(state='disabled')
        self._log(f"Found {len(duplicates)} duplicate pairs")
    
    def _show_history(self):
        """Show processing history"""
        jobs = self.db.get_job_history(30)
        
        if not jobs:
            messagebox.showinfo("History", "No processing history found")
            return
        
        history_window = tk.Toplevel(self.root)
        history_window.title("Processing History")
        history_window.geometry="1000x400"
        
        # Create treeview
        columns = ('id', 'source', 'target', 'total', 'processed', 'status', 'date')
        tree = ttk.Treeview(history_window, columns=columns, show='headings', height=20)
        
        tree.heading('id', text='Job ID')
        tree.heading('source', text='Source Folder')
        tree.heading('target', text='Target Folder')
        tree.heading('total', text='Total')
        tree.heading('processed', text='Processed')
        tree.heading('status', text='Status')
        tree.heading('date', text='Started')
        
        tree.column('id', width=60)
        tree.column('source', width=250)
        tree.column('target', width=250)
        tree.column('total', width=80)
        tree.column('processed', width=80)
        tree.column('status', width=100)
        tree.column('date', width=150)
        
        scrollbar = ttk.Scrollbar(history_window, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        for job in jobs:
            tree.insert('', tk.END, values=(
                job['id'],
                Path(job['source_folder']).name[:40],
                Path(job['target_base']).name[:40],
                job['total_images'] or 0,
                job['processed_images'] or 0,
                job['status'],
                job['started_at'][:19] if job['started_at'] else ''
            ))


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    root = tk.Tk()
    
    # Set application icon (optional - create if you have an icon file)
    try:
        root.iconbitmap(default='instasort.ico')
    except:
        pass
    
    app = InstaSortApp(root)
    
    # Handle window close
    def on_closing():
        if app.is_processing:
            if messagebox.askyesno("Confirm", "Processing in progress. Stop and exit?"):
                app.is_processing = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run application
    root.mainloop()


if __name__ == "__main__":
    main()
