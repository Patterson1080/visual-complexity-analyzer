import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QFileDialog, 
                             QProgressBar, QGroupBox, QFormLayout, QSpinBox, 
                             QComboBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import json
from src.workers import AnalysisThread

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Fractal Dimensionality Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        self.current_video_path = None
        self.analysis_thread = None
        self.results_data = []
        self.batch_queue = []
        self.is_batch_mode = False
        
        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Top Controls
        self.create_controls()
        
        # Middle Splitter (Settings + Plots)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        
        # Left Panel: Settings
        self.settings_panel = QGroupBox("Analysis Settings")
        self.create_settings_panel()
        self.splitter.addWidget(self.settings_panel)
        
        # Right Panel: Plots
        self.plot_panel = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_panel)
        self.create_plots()
        self.splitter.addWidget(self.plot_panel)
        
        # Set initial splitter sizes
        self.splitter.setSizes([300, 900])
        
        # Bottom Status
        self.progress_bar = QProgressBar()
        self.main_layout.addWidget(self.progress_bar)
        
    def create_controls(self):
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton("Load Video")
        self.btn_load.clicked.connect(self.load_video)
        control_layout.addWidget(self.btn_load)
        
        self.lbl_file = QLabel("No file loaded")
        control_layout.addWidget(self.lbl_file)
        
        self.btn_start = QPushButton("Start Analysis")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)
        control_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        self.btn_batch = QPushButton("Batch Process Folder")
        self.btn_batch.clicked.connect(self.batch_process)
        control_layout.addWidget(self.btn_batch)

        self.btn_export = QPushButton("Export Results")
        self.btn_export.clicked.connect(self.export_results)
        self.btn_export.setEnabled(False)
        control_layout.addWidget(self.btn_export)
        
        self.main_layout.addLayout(control_layout)

    def create_settings_panel(self):
        layout = QFormLayout()
        
        self.spin_sampling = QSpinBox()
        self.spin_sampling.setRange(1, 1000)
        self.spin_sampling.setValue(1)
        layout.addRow("Sampling Rate (Every N frames):", self.spin_sampling)
        
        self.combo_method = QComboBox()
        self.combo_method.addItems(["canny", "sobel"])
        layout.addRow("Edge Method:", self.combo_method)
        
        self.combo_threshold = QComboBox()
        self.combo_threshold.addItems(["auto", "manual"])
        layout.addRow("Threshold Mode:", self.combo_threshold)
        
        self.spin_blur = QSpinBox()
        self.spin_blur.setRange(0, 21)
        self.spin_blur.setValue(5)
        self.spin_blur.setSingleStep(2) # Odd numbers only ideally
        layout.addRow("Blur Kernel Size:", self.spin_blur)
        
        self.combo_analysis = QComboBox()
        self.combo_analysis.addItems(["Edge + Box Counting", "Differential Box Counting (DBC)", "Fourier Slope"])
        self.combo_analysis.currentIndexChanged.connect(self.toggle_edge_settings)
        layout.addRow("Analysis Method:", self.combo_analysis)
        
        self.settings_panel.setLayout(layout)

    def toggle_edge_settings(self):
        # Disable edge settings if not Box Counting
        method = self.combo_analysis.currentText()
        is_edge = "Edge" in method
        
        self.combo_method.setEnabled(is_edge)
        self.combo_threshold.setEnabled(is_edge)
        self.spin_blur.setEnabled(is_edge)

    def create_plots(self):
        # Tabs for "Analysis" and "Summary"
        self.tabs = QTabWidget()
        self.plot_layout.addWidget(self.tabs)
        
        # --- Tab 1: Analysis ---
        self.tab_analysis = QWidget()
        analysis_layout = QVBoxLayout(self.tab_analysis)
        
        # Top: Video Preview | Edge Preview | Log-Log Plot
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        
        # Video Preview
        self.lbl_frame = QLabel("Original Frame")
        self.lbl_frame.setFixedSize(320, 240)
        self.lbl_frame.setStyleSheet("border: 1px solid black; background-color: #eee;")
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.lbl_frame)
        
        # Edge Preview
        self.lbl_edges = QLabel("Edge Detection")
        self.lbl_edges.setFixedSize(320, 240)
        self.lbl_edges.setStyleSheet("border: 1px solid black; background-color: #eee;")
        self.lbl_edges.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.lbl_edges)
        
        # Log-Log Plot
        self.fig_log = Figure(figsize=(4, 3), dpi=100)
        self.canvas_log = FigureCanvas(self.fig_log)
        self.ax_log = self.fig_log.add_subplot(111)
        self.ax_log.set_title("Log-Log Plot")
        self.ax_log.set_xlabel("log(1/s)")
        self.ax_log.set_ylabel("log(N(s))")
        self.ax_log.grid(True)
        top_layout.addWidget(self.canvas_log)
        
        analysis_layout.addWidget(top_widget)
        
        # Bottom: D(t) Plot
        self.fig_time = Figure(figsize=(8, 4), dpi=100)
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        self.ax_time.set_title("Fractal Dimension D(t)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("D")
        self.ax_time.grid(True)
        analysis_layout.addWidget(self.canvas_time)
        
        self.tabs.addTab(self.tab_analysis, "Real-time Analysis")
        
        # --- Tab 2: Summary ---
        self.tab_summary = QWidget()
        summary_layout = QVBoxLayout(self.tab_summary)
        
        # Stats Table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(self.stats_table)
        
        # Histogram
        self.fig_hist = Figure(figsize=(8, 4), dpi=100)
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.ax_hist.set_title("Distribution of D values")
        self.ax_hist.set_xlabel("D")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.grid(True)
        summary_layout.addWidget(self.canvas_hist)
        
        self.tabs.addTab(self.tab_summary, "Summary & Statistics")

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.current_video_path = path
            self.lbl_file.setText(os.path.basename(path))
            self.btn_start.setEnabled(True)
            self.btn_batch.setEnabled(True)
            self.results_data = []
            
            # Clear plots
            self.ax_time.clear()
            self.ax_log.clear()
            self.ax_hist.clear()
            self.setup_axes()
            self.canvas_time.draw()
            self.canvas_log.draw()
            self.canvas_hist.draw()

    def setup_axes(self):
        self.ax_time.set_title("Fractal Dimension D(t)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("D")
        self.ax_time.grid(True)
        
        self.ax_log.set_title("Log-Log Plot")
        self.ax_log.set_xlabel("log(1/s)")
        self.ax_log.set_ylabel("log(N(s))")
        self.ax_log.grid(True)
        
        self.ax_hist.set_title("Distribution of D values")
        self.ax_hist.set_xlabel("D")
        self.ax_hist.set_ylabel("Count")
        self.ax_hist.grid(True)

    def start_analysis(self):
        if not self.current_video_path:
            return
            
        analysis_map = {
            "Edge + Box Counting": "box_counting",
            "Differential Box Counting (DBC)": "dbc",
            "Fourier Slope": "fourier"
        }
            
        settings = {
            'sampling_rate': self.spin_sampling.value(),
            'edge_method': self.combo_method.currentText(),
            'threshold_mode': self.combo_threshold.currentText(),
            'blur_kernel_size': self.spin_blur.value(),
            'analysis_type': analysis_map.get(self.combo_analysis.currentText(), 'box_counting')
        }
        
        self.analysis_thread = AnalysisThread(self.current_video_path, settings)
        self.analysis_thread.progress_updated.connect(self.update_progress)
        self.analysis_thread.frame_processed.connect(self.update_plots)
        self.analysis_thread.analysis_finished.connect(self.analysis_finished)
        
        self.analysis_thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_load.setEnabled(False)
        self.btn_batch.setEnabled(False)
        self.results_data = []

    def stop_analysis(self):
        if self.analysis_thread:
            self.analysis_thread.stop()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def update_plots(self, result):
        self.results_data.append(result)
        
        # Update D(t) plot
        timestamps = [r['timestamp'] for r in self.results_data]
        Ds = [r['D'] for r in self.results_data]
        
        self.ax_time.clear()
        self.ax_time.plot(timestamps, Ds, 'b-')
        self.ax_time.set_title("Fractal Dimension D(t)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("D")
        self.ax_time.grid(True)
        self.canvas_time.draw()
        
        # Update Log-Log plot
        scales = result['scales']
        counts = result['counts']
        # Check if arrays are not empty. Use len() as they might be numpy arrays.
        if len(scales) > 0 and len(counts) > 0:
            self.ax_log.clear()
            self.ax_log.plot(scales, counts, 'ro-')
            reliability = "" if result.get('reliable', True) else " [UNRELIABLE]"
            self.ax_log.set_title(f"Log-Log (D={result['D']:.2f}, R²={result['R2']:.2f}){reliability}")
            
            method = result.get('method', 'box_counting')
            if method == 'fourier':
                self.ax_log.set_xlabel("log(Frequency)")
                self.ax_log.set_ylabel("log(Power)")
            else:
                self.ax_log.set_xlabel("log(1/s)")
                self.ax_log.set_ylabel("log(N(s))")
                
            self.ax_log.grid(True)
            self.canvas_log.draw()

        # Update Images (Preview)
        frame = result.get('frame')
        edges = result.get('edges')
        
        if frame is not None:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.lbl_frame.setPixmap(QPixmap.fromImage(qt_image).scaled(self.lbl_frame.size(), Qt.KeepAspectRatio))
            
        if edges is not None:
            # Edges is grayscale
            h, w = edges.shape
            bytes_per_line = w
            qt_image = QImage(edges.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            self.lbl_edges.setPixmap(QPixmap.fromImage(qt_image).scaled(self.lbl_edges.size(), Qt.KeepAspectRatio))

        # Update Statistics & Histogram (every 5 frames)
        if len(self.results_data) % 5 == 0:
            self.update_stats(Ds)

    def update_stats(self, Ds):
        if not Ds:
            return
            
        # Histogram
        self.ax_hist.clear()
        self.ax_hist.hist(Ds, bins=20, color='skyblue', edgecolor='black')
        self.ax_hist.axvline(1.3, color='r', linestyle='--', label='Optimal Low (1.3)')
        self.ax_hist.axvline(1.5, color='r', linestyle='--', label='Optimal High (1.5)')
        self.ax_hist.set_title("Distribution of D values")
        self.ax_hist.legend()
        self.canvas_hist.draw()
        
        # Stats Table
        s = pd.Series(Ds)
        stats = {
            "Mean D": f"{s.mean():.4f}",
            "Median D": f"{s.median():.4f}",
            "Std Dev": f"{s.std():.4f}",
            "Min D": f"{s.min():.4f}",
            "Max D": f"{s.max():.4f}",
            "% in 1.3-1.5": f"{((s >= 1.3) & (s <= 1.5)).mean() * 100:.1f}%"
        }
        
        self.stats_table.setRowCount(len(stats))
        for i, (k, v) in enumerate(stats.items()):
            self.stats_table.setItem(i, 0, QTableWidgetItem(k))
            self.stats_table.setItem(i, 1, QTableWidgetItem(v))

    def batch_process(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Process")
        if not folder:
            return
            
        # Find video files
        extensions = ('.mp4', '.avi', '.mov', '.mkv')
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(extensions)]
        
        if not files:
            return
            
        self.batch_queue = files
        self.is_batch_mode = True
        self.lbl_file.setText(f"Batch Processing {len(files)} files...")
        self.process_next_in_queue()

    def process_next_in_queue(self):
        if not self.batch_queue:
            self.is_batch_mode = False
            self.lbl_file.setText("Batch Processing Complete")
            self.btn_start.setEnabled(True)
            self.btn_load.setEnabled(True)
            self.btn_batch.setEnabled(True)
            return
            
        next_file = self.batch_queue.pop(0)
        self.current_video_path = next_file
        self.lbl_file.setText(f"Processing: {os.path.basename(next_file)} ({len(self.batch_queue)} remaining)")
        
        # Start analysis for this file
        self.start_analysis()

    def analysis_finished(self):
        # Auto-export if in batch mode
        if self.is_batch_mode and self.current_video_path and self.results_data:
            # Generate filename
            base = os.path.splitext(os.path.basename(self.current_video_path))[0]
            folder = os.path.dirname(self.current_video_path)
            csv_path = os.path.join(folder, f"fractal_analysis_{base}.csv")
            
            try:
                df = pd.DataFrame(self.results_data)
                df.to_csv(csv_path, index=False)
                # Also save plot
                plot_path = os.path.join(folder, f"fractal_timeseries_{base}.png")
                self.fig_time.savefig(plot_path)
                
                # Save JSON Summary
                json_path = os.path.join(folder, f"fractal_summary_{base}.json")
                s = df['D']
                summary = {
                    "mean_D": float(s.mean()),
                    "median_D": float(s.median()),
                    "std_D": float(s.std()),
                    "min_D": float(s.min()),
                    "max_D": float(s.max()),
                    "percent_optimal": float(((s >= 1.3) & (s <= 1.5)).mean() * 100),
                    "total_frames": len(s),
                    "video_path": self.current_video_path
                }
                
                with open(json_path, 'w') as f:
                    json.dump(summary, f, indent=4)
                    
            except Exception as e:
                print(f"Error saving batch results for {base}: {e}")
                
            # Trigger next
            self.process_next_in_queue()
        else:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.btn_load.setEnabled(True)
            self.btn_export.setEnabled(True)
            self.btn_batch.setEnabled(True)
            self.progress_bar.setValue(self.progress_bar.maximum())

    def export_results(self):
        if not self.results_data:
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.results_data)
            df.to_csv(path, index=False)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
