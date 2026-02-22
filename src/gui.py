import sys
import os
import ctypes
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QHBoxLayout, QWidget, QPushButton, QFileDialog,
                             QProgressBar, QGroupBox, QFormLayout, QSpinBox,
                             QDoubleSpinBox, QSlider, QTimeEdit,
                             QComboBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt5.QtCore import Qt, QTime
from PyQt5.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import numpy as np
import json
from src.workers import AnalysisThread
from src.core import GPU_AVAILABLE

# --- Dark Theme Colors ---
BG_DARK = "#1a1a2e"
BG_SURFACE = "#2a2a3e"
BG_INPUT = "#16213e"
ACCENT = "#d4a574"
TEXT_PRIMARY = "#e8e8e8"
TEXT_SECONDARY = "#888888"
BORDER = "#333333"
ERROR_RED = "#e74c3c"
SUCCESS_GREEN = "#2ecc71"

DARK_STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRIMARY};
}}

QPushButton {{
    background-color: transparent;
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 8px;
    padding: 8px 20px;
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {BG_SURFACE};
}}
QPushButton:disabled {{
    color: {TEXT_SECONDARY};
    border-color: {BORDER};
    background-color: transparent;
}}
QPushButton#primary {{
    background-color: {ACCENT};
    color: {BG_DARK};
    border: none;
}}
QPushButton#primary:hover {{
    background-color: #e0b68a;
}}
QPushButton#primary:disabled {{
    background-color: {BORDER};
    color: {TEXT_SECONDARY};
}}

QGroupBox {{
    background-color: {BG_SURFACE};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 16px;
    margin-top: 12px;
    font-weight: bold;
    color: {TEXT_SECONDARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    padding: 0 8px;
    color: {TEXT_SECONDARY};
}}

QComboBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {TEXT_SECONDARY};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    selection-background-color: {BG_SURFACE};
    selection-color: {ACCENT};
}}

QSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}}
QSpinBox:hover {{
    border-color: {ACCENT};
}}
QSpinBox::up-button, QSpinBox::down-button {{
    background-color: {BG_SURFACE};
    border: none;
    width: 16px;
}}
QSpinBox::up-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_SECONDARY};
}}
QSpinBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SECONDARY};
}}

QDoubleSpinBox {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
}}
QDoubleSpinBox:hover {{
    border-color: {ACCENT};
}}
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {BG_SURFACE};
    border: none;
    width: 16px;
}}
QDoubleSpinBox::up-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_SECONDARY};
}}
QDoubleSpinBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SECONDARY};
}}

QSlider::groove:horizontal {{
    background-color: {BG_INPUT};
    border: 1px solid {BORDER};
    border-radius: 3px;
    height: 6px;
}}
QSlider::handle:horizontal {{
    background-color: {ACCENT};
    border: none;
    border-radius: 6px;
    width: 12px;
    margin: -4px 0;
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

QTimeEdit {{
    background-color: {BG_INPUT};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 24px;
    font-family: monospace;
    font-size: 13px;
}}
QTimeEdit:hover {{
    border-color: {ACCENT};
}}
QTimeEdit::up-button, QTimeEdit::down-button {{
    background-color: {BG_SURFACE};
    border: none;
    width: 16px;
}}
QTimeEdit::up-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-bottom: 5px solid {TEXT_SECONDARY};
}}
QTimeEdit::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SECONDARY};
}}

QTabWidget::pane {{
    background-color: {BG_DARK};
    border: 1px solid {BORDER};
    border-top: none;
}}
QTabBar::tab {{
    background-color: {BG_SURFACE};
    color: {TEXT_SECONDARY};
    padding: 8px 20px;
    border: none;
    border-bottom: 2px solid transparent;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    background-color: {BG_DARK};
}}
QTabBar::tab:hover {{
    color: {TEXT_PRIMARY};
    background-color: {BG_DARK};
}}

QProgressBar {{
    background-color: {BG_SURFACE};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 4px;
}}

QTableWidget {{
    background-color: {BG_DARK};
    alternate-background-color: {BG_SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    gridline-color: transparent;
    selection-background-color: {BG_SURFACE};
    selection-color: {ACCENT};
}}
QTableWidget::item {{
    padding: 4px 8px;
    border-bottom: 1px solid {BORDER};
}}
QHeaderView::section {{
    background-color: {BG_SURFACE};
    color: {TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: bold;
}}

QLabel {{
    background-color: transparent;
    color: {TEXT_PRIMARY};
}}

NavigationToolbar2QT {{
    background-color: {BG_SURFACE};
    border: none;
    spacing: 4px;
}}
NavigationToolbar2QT > QToolButton {{
    background-color: transparent;
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 2px;
}}
NavigationToolbar2QT > QToolButton:hover {{
    background-color: {BG_DARK};
    border-color: {ACCENT};
}}

QSplitter::handle {{
    background-color: {BORDER};
    width: 2px;
}}

QScrollBar:vertical {{
    background-color: {BG_DARK};
    width: 10px;
    border: none;
}}
QScrollBar::handle:vertical {{
    background-color: {BORDER};
    border-radius: 5px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QScrollBar:horizontal {{
    background-color: {BG_DARK};
    height: 10px;
    border: none;
}}
QScrollBar::handle:horizontal {{
    background-color: {BORDER};
    border-radius: 5px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0px;
}}

QFormLayout {{
    color: {TEXT_PRIMARY};
}}
"""

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
        self.btn_load.setObjectName("primary")
        self.btn_load.clicked.connect(self.load_video)
        control_layout.addWidget(self.btn_load)

        self.lbl_file = QLabel("No file loaded")
        control_layout.addWidget(self.lbl_file)

        self.btn_start = QPushButton("Start Analysis")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self.start_analysis)
        self.btn_start.setEnabled(False)
        control_layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("secondary")
        self.btn_stop.clicked.connect(self.stop_analysis)
        self.btn_stop.setEnabled(False)
        control_layout.addWidget(self.btn_stop)

        self.btn_batch = QPushButton("Batch Process Folder")
        self.btn_batch.setObjectName("secondary")
        self.btn_batch.clicked.connect(self.batch_process)
        control_layout.addWidget(self.btn_batch)

        self.btn_export = QPushButton("Export Results")
        self.btn_export.setObjectName("secondary")
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

        # Clip range: HH:MM:SS → HH:MM:SS (like VLC / media players)
        clip_widget = QWidget()
        clip_layout = QHBoxLayout(clip_widget)
        clip_layout.setContentsMargins(0, 0, 0, 0)
        clip_layout.setSpacing(6)

        self.time_clip_start = QTimeEdit()
        self.time_clip_start.setDisplayFormat("HH:mm:ss")
        self.time_clip_start.setTime(QTime(0, 0, 0))
        self.time_clip_start.setToolTip("Start analysis from this timestamp")

        arrow_lbl = QLabel("→")
        arrow_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; font-size: 14px;")

        self.time_clip_end = QTimeEdit()
        self.time_clip_end.setDisplayFormat("HH:mm:ss")
        self.time_clip_end.setTime(QTime(0, 0, 0))
        self.time_clip_end.setToolTip(
            "Stop analysis at this timestamp. Set to 00:00:00 to analyze to end of video.")

        clip_layout.addWidget(self.time_clip_start)
        clip_layout.addWidget(arrow_lbl)
        clip_layout.addWidget(self.time_clip_end)
        clip_layout.addStretch()
        layout.addRow("Clip Range:", clip_widget)

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
        self.combo_analysis.addItems(["Moisy Threshold + Box Counting",
                                       "Edge + Box Counting",
                                       "Differential Box Counting (DBC)",
                                       "Fourier Slope"])
        self.combo_analysis.currentIndexChanged.connect(self.toggle_edge_settings)
        layout.addRow("Analysis Method:", self.combo_analysis)

        # --- Moisy-specific settings ---
        # Binarization threshold slider + spinbox
        moisy_thresh_widget = QWidget()
        moisy_thresh_layout = QHBoxLayout(moisy_thresh_widget)
        moisy_thresh_layout.setContentsMargins(0, 0, 0, 0)

        self.slider_moisy_thresh = QSlider(Qt.Horizontal)
        self.slider_moisy_thresh.setRange(1, 99)  # 0.01–0.99
        self.slider_moisy_thresh.setValue(25)      # 0.25
        self.slider_moisy_thresh.setToolTip(
            "Threshold for grayscale\u2192binary conversion (0\u20131 scale). "
            "Default 0.25 matches published method.")

        self.spin_moisy_thresh = QDoubleSpinBox()
        self.spin_moisy_thresh.setRange(0.01, 0.99)
        self.spin_moisy_thresh.setSingleStep(0.01)
        self.spin_moisy_thresh.setDecimals(2)
        self.spin_moisy_thresh.setValue(0.25)
        self.spin_moisy_thresh.setToolTip(self.slider_moisy_thresh.toolTip())

        # Keep slider and spinbox in sync
        self.slider_moisy_thresh.valueChanged.connect(
            lambda v: self.spin_moisy_thresh.setValue(v / 100.0))
        self.spin_moisy_thresh.valueChanged.connect(
            lambda v: self.slider_moisy_thresh.setValue(int(v * 100)))

        moisy_thresh_layout.addWidget(self.slider_moisy_thresh, stretch=1)
        moisy_thresh_layout.addWidget(self.spin_moisy_thresh)
        self.lbl_moisy_thresh = QLabel("Binarization Threshold:")
        layout.addRow(self.lbl_moisy_thresh, moisy_thresh_widget)

        # Scale range spinboxes
        scale_range_widget = QWidget()
        scale_range_layout = QHBoxLayout(scale_range_widget)
        scale_range_layout.setContentsMargins(0, 0, 0, 0)

        self.spin_scale_start = QSpinBox()
        self.spin_scale_start.setRange(1, 15)
        self.spin_scale_start.setValue(4)
        self.spin_scale_start.setToolTip(
            "MATLAB-indexed range of local slopes to average. "
            "Default 4\u20138 uses mid-range scales.")

        self.spin_scale_end = QSpinBox()
        self.spin_scale_end.setRange(1, 15)
        self.spin_scale_end.setValue(8)
        self.spin_scale_end.setToolTip(self.spin_scale_start.toolTip())

        scale_range_layout.addWidget(QLabel("Start:"))
        scale_range_layout.addWidget(self.spin_scale_start)
        scale_range_layout.addWidget(QLabel("End:"))
        scale_range_layout.addWidget(self.spin_scale_end)
        self.lbl_scale_range = QLabel("Scale Range:")
        layout.addRow(self.lbl_scale_range, scale_range_widget)

        # Trigger initial visibility
        self.toggle_edge_settings()

        # GPU / CPU status indicator
        if GPU_AVAILABLE:
            self.lbl_compute = QLabel("Compute: GPU (CUDA)")
            self.lbl_compute.setStyleSheet(f"color: {SUCCESS_GREEN}; font-weight: bold;")
        else:
            self.lbl_compute = QLabel("Compute: CPU")
            self.lbl_compute.setStyleSheet(f"color: {TEXT_SECONDARY}; font-weight: bold;")
        layout.addRow(self.lbl_compute)

        self.settings_panel.setLayout(layout)

    def toggle_edge_settings(self):
        method = self.combo_analysis.currentText()
        is_edge = "Edge" in method
        is_moisy = "Moisy" in method

        # Edge-specific controls
        self.combo_method.setEnabled(is_edge)
        self.combo_threshold.setEnabled(is_edge)
        self.spin_blur.setEnabled(is_edge)

        # Moisy-specific controls
        self.slider_moisy_thresh.setVisible(is_moisy)
        self.spin_moisy_thresh.setVisible(is_moisy)
        self.lbl_moisy_thresh.setVisible(is_moisy)
        self.spin_scale_start.setVisible(is_moisy)
        self.spin_scale_end.setVisible(is_moisy)
        self.lbl_scale_range.setVisible(is_moisy)
        # Also hide the parent widgets of scale range
        self.spin_scale_start.parent().setVisible(is_moisy)
        self.spin_moisy_thresh.parent().setVisible(is_moisy)

    def _style_figure(self, fig, ax):
        """Apply dark theme to a matplotlib figure and axes."""
        fig.patch.set_facecolor(BG_SURFACE)
        ax.set_facecolor(BG_DARK)
        ax.tick_params(colors=TEXT_PRIMARY)
        ax.xaxis.label.set_color(TEXT_PRIMARY)
        ax.yaxis.label.set_color(TEXT_PRIMARY)
        ax.title.set_color(TEXT_PRIMARY)
        for spine in ax.spines.values():
            spine.set_color(BORDER)
        ax.grid(True, color='#444444', alpha=0.5)

    def _save_fig_publication(self, fig, path, dpi=300):
        """Save a figure with white background and black text for publication."""
        # Store original colors
        orig_fig_color = fig.get_facecolor()
        orig_ax_colors = []
        for ax in fig.get_axes():
            orig_ax_colors.append({
                'facecolor': ax.get_facecolor(),
                'title_color': ax.title.get_color(),
                'xlabel_color': ax.xaxis.label.get_color(),
                'ylabel_color': ax.yaxis.label.get_color(),
                'tick_colors': (ax.xaxis.get_tick_params(), ax.yaxis.get_tick_params()),
                'spine_colors': {k: s.get_edgecolor() for k, s in ax.spines.items()},
                'grid_lines': ax.get_xgridlines() + ax.get_ygridlines(),
            })
            # Apply publication style
            ax.set_facecolor('white')
            ax.title.set_color('black')
            ax.xaxis.label.set_color('black')
            ax.yaxis.label.set_color('black')
            ax.tick_params(colors='black')
            for spine in ax.spines.values():
                spine.set_color('black')
            ax.grid(True, color='#cccccc', alpha=0.5)
            # Fix legend colors if present
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('#cccccc')
                for text in legend.get_texts():
                    text.set_color('black')

        fig.patch.set_facecolor('white')
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')

        # Restore original dark theme colors
        fig.patch.set_facecolor(orig_fig_color)
        for ax, orig in zip(fig.get_axes(), orig_ax_colors):
            ax.set_facecolor(orig['facecolor'])
            ax.title.set_color(orig['title_color'])
            ax.xaxis.label.set_color(orig['xlabel_color'])
            ax.yaxis.label.set_color(orig['ylabel_color'])
            ax.tick_params(axis='x', colors=TEXT_PRIMARY)
            ax.tick_params(axis='y', colors=TEXT_PRIMARY)
            for k, s in ax.spines.items():
                s.set_edgecolor(orig['spine_colors'][k])
            ax.grid(True, color='#444444', alpha=0.5)
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(BG_SURFACE)
                legend.get_frame().set_edgecolor(BORDER)
                for text in legend.get_texts():
                    text.set_color(TEXT_PRIMARY)

        # Redraw to restore on-screen appearance
        fig.canvas.draw()

    def _on_time_interact(self, event):
        """Mark that the user has manually panned/zoomed the D(t) plot."""
        if event.inaxes == self.ax_time:
            self._time_user_interacted = True

    def _reset_time_view(self):
        """Reset auto-scrolling on the D(t) plot."""
        self._time_user_interacted = False

    def create_plots(self):
        # Tabs for "Analysis" and "Summary"
        self.tabs = QTabWidget()
        self.plot_layout.addWidget(self.tabs)

        # --- Tab 1: Analysis ---
        self.tab_analysis = QWidget()
        analysis_layout = QVBoxLayout(self.tab_analysis)

        # Left column: Video previews stacked vertically
        # Right column: Plots
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)

        # Left: Video previews (top/bottom)
        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)

        # Original Frame (top)
        self.lbl_frame = QLabel("Original Frame")
        self.lbl_frame.setMinimumSize(400, 300)
        self.lbl_frame.setStyleSheet(f"border: 1px solid {BORDER}; background-color: #111; border-radius: 6px;")
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        self.lbl_frame.setScaledContents(False)
        preview_layout.addWidget(self.lbl_frame)

        # Edge Detection (bottom)
        self.lbl_edges = QLabel("Edge Detection")
        self.lbl_edges.setMinimumSize(400, 300)
        self.lbl_edges.setStyleSheet(f"border: 1px solid {BORDER}; background-color: #111; border-radius: 6px;")
        self.lbl_edges.setAlignment(Qt.AlignCenter)
        self.lbl_edges.setScaledContents(False)
        preview_layout.addWidget(self.lbl_edges)

        content_layout.addWidget(preview_widget, stretch=1)

        # Right: Plots stacked vertically
        plots_widget = QWidget()
        plots_layout = QVBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)

        # Log-Log Plot
        self.fig_log = Figure(figsize=(4, 3), dpi=100)
        self.canvas_log = FigureCanvas(self.fig_log)
        self.ax_log = self.fig_log.add_subplot(111)
        self.ax_log.set_title("Log-Log Plot")
        self.ax_log.set_xlabel("log(1/s)")
        self.ax_log.set_ylabel("log(N(s))")
        self._style_figure(self.fig_log, self.ax_log)
        self.line_log, = self.ax_log.plot([], [], 'o-', color=ERROR_RED, markersize=4)
        self.toolbar_log = NavigationToolbar(self.canvas_log, self)
        self.toolbar_log.setStyleSheet(f"background-color: {BG_SURFACE}; border: none;")
        plots_layout.addWidget(self.toolbar_log)
        plots_layout.addWidget(self.canvas_log)

        # D(t) Plot
        self.fig_time = Figure(figsize=(4, 3), dpi=100)
        self.canvas_time = FigureCanvas(self.fig_time)
        self.ax_time = self.fig_time.add_subplot(111)
        self.ax_time.set_title("Fractal Dimension D(t)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylabel("D")
        self._style_figure(self.fig_time, self.ax_time)
        self.ax_time.set_ylim(0.5, 2.5)
        self.ax_time.set_xlim(0, 30)
        self.line_time, = self.ax_time.plot([], [], color=ACCENT, linewidth=1.5)
        self._time_user_interacted = False
        self._time_window = 30  # seconds of visible window
        self.toolbar_time = NavigationToolbar(self.canvas_time, self)
        self.toolbar_time.setStyleSheet(f"background-color: {BG_SURFACE}; border: none;")
        # Track when user manually pans/zooms the D(t) plot
        self.canvas_time.mpl_connect('button_press_event', self._on_time_interact)
        self.canvas_time.mpl_connect('scroll_event', self._on_time_interact)
        plots_layout.addWidget(self.toolbar_time)
        plots_layout.addWidget(self.canvas_time)

        content_layout.addWidget(plots_widget, stretch=1)
        analysis_layout.addWidget(content_widget)

        self.tabs.addTab(self.tab_analysis, "Real-time Analysis")

        # --- Tab 2: Summary ---
        self.tab_summary = QWidget()
        summary_layout = QVBoxLayout(self.tab_summary)

        # Stats Table
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(2)
        self.stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.stats_table.setAlternatingRowColors(True)
        summary_layout.addWidget(self.stats_table)

        # Histogram
        self.fig_hist = Figure(figsize=(8, 4), dpi=100)
        self.canvas_hist = FigureCanvas(self.fig_hist)
        self.ax_hist = self.fig_hist.add_subplot(111)
        self.ax_hist.set_title("Distribution of D values")
        self.ax_hist.set_xlabel("D")
        self.ax_hist.set_ylabel("Count")
        self._style_figure(self.fig_hist, self.ax_hist)
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

            # Set clip range from video duration
            _cap = cv2.VideoCapture(path)
            if _cap.isOpened():
                _fps = _cap.get(cv2.CAP_PROP_FPS)
                _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                _dur = int(_total / _fps) if _fps > 0 else 0
                _cap.release()
                self.time_clip_start.setTime(QTime(0, 0, 0))
                self.time_clip_end.setTime(QTime(_dur // 3600, (_dur % 3600) // 60, _dur % 60))

            # Clear plots
            self.ax_hist.clear()
            self.ax_hist.set_title("Distribution of D values")
            self.ax_hist.set_xlabel("D")
            self.ax_hist.set_ylabel("Count")
            self._style_figure(self.fig_hist, self.ax_hist)
            self.canvas_hist.draw()

            # Reset D(t) and log-log lines
            self.line_time.set_data([], [])
            self.ax_time.set_xlim(0, self._time_window)
            self.ax_time.set_ylim(0.5, 2.5)
            self._time_user_interacted = False
            self.canvas_time.draw()

            self.line_log.set_data([], [])
            self.ax_log.relim()
            self.canvas_log.draw()

    @staticmethod
    def _qtime_to_sec(t):
        return t.hour() * 3600 + t.minute() * 60 + t.second()

    def start_analysis(self):
        if not self.current_video_path:
            return

        analysis_map = {
            "Moisy Threshold + Box Counting": "moisy_boxcount",
            "Edge + Box Counting": "box_counting",
            "Differential Box Counting (DBC)": "dbc",
            "Fourier Slope": "fourier"
        }

        settings = {
            'sampling_rate': self.spin_sampling.value(),
            'edge_method': self.combo_method.currentText(),
            'threshold_mode': self.combo_threshold.currentText(),
            'blur_kernel_size': self.spin_blur.value(),
            'analysis_type': analysis_map.get(self.combo_analysis.currentText(), 'moisy_boxcount'),
            'moisy_threshold': self.spin_moisy_thresh.value(),
            'scale_range': (self.spin_scale_start.value(), self.spin_scale_end.value()),
            'clip_start_sec': self._qtime_to_sec(self.time_clip_start.time()),
            'clip_end_sec': self._qtime_to_sec(self.time_clip_end.time()),
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
        self._time_user_interacted = False

    def stop_analysis(self):
        if self.analysis_thread:
            self.analysis_thread.stop()

    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)

    def update_plots(self, result):
        # Store only numeric data (not images) to prevent memory leak
        frame = result.pop('frame', None)
        edges = result.pop('edges', None)
        result.pop('df', None)  # local slopes array (not needed in CSV)
        self.results_data.append(result)

        # Throttle plot updates — only redraw plots every 3 frames
        should_redraw_plots = len(self.results_data) % 3 == 0 or len(self.results_data) == 1

        if should_redraw_plots:
            # Update D(t) plot
            timestamps = [r['timestamp'] for r in self.results_data]
            Ds = [r['D'] for r in self.results_data]

            self.line_time.set_data(timestamps, Ds)
            # Auto-scroll sliding window unless user has manually panned/zoomed
            if not self._time_user_interacted:
                t_now = timestamps[-1]
                if t_now > self._time_window:
                    self.ax_time.set_xlim(t_now - self._time_window, t_now + 2)
                else:
                    self.ax_time.set_xlim(0, self._time_window)
            self.canvas_time.draw()

            # Update Log-Log plot
            scales = result['scales']
            counts = result['counts']
            method = result.get('method', 'box_counting')
            # Check if arrays are not empty. Use len() as they might be numpy arrays.
            if len(scales) > 0 and len(counts) > 0:
                self.ax_log.clear()
                self._style_figure(self.fig_log, self.ax_log)

                if method == 'moisy_boxcount':
                    # Moisy: log(R) vs log(N), highlight selected scale range
                    self.ax_log.plot(scales, counts, 'o-', color=ERROR_RED, markersize=4)

                    # Highlight the scale-range points used for D
                    sr = result.get('scale_range', '4-8')
                    parts = sr.split('-')
                    lo = int(parts[0]) - 1   # MATLAB→Python
                    hi = int(parts[1])
                    # The scale/count arrays have length p+1; indices lo..hi-1 in
                    # the df array correspond to the *intervals* between adjacent
                    # (scale, count) points.  Highlight points lo..hi (inclusive).
                    lo_pt = max(0, lo)
                    hi_pt = min(len(scales), hi + 1)
                    self.ax_log.plot(scales[lo_pt:hi_pt], counts[lo_pt:hi_pt],
                                    's', color=ACCENT, markersize=8, zorder=5,
                                    label=f"Scales {parts[0]}\u2013{parts[1]}")

                    D = result['D']
                    D_std = result.get('D_std', 0)
                    title_text = f"Log-Log  D = {D:.4f} \u00b1 {D_std:.4f}"
                    self.ax_log.set_title(title_text, color=TEXT_PRIMARY)
                    self.ax_log.set_xlabel("log(R)")
                    self.ax_log.set_ylabel("log(N)")
                    self.ax_log.legend(facecolor=BG_SURFACE, edgecolor=BORDER,
                                       labelcolor=TEXT_PRIMARY, fontsize='small')
                else:
                    self.ax_log.plot(scales, counts, 'o-', color=ERROR_RED, markersize=4)
                    reliability = "" if result.get('reliable', True) else " [UNRELIABLE]"
                    title_text = f"Log-Log (D={result['D']:.2f}, R\u00b2={result['R2']:.2f})"
                    if reliability:
                        self.ax_log.set_title(title_text + reliability, color=ERROR_RED)
                    else:
                        self.ax_log.set_title(title_text, color=TEXT_PRIMARY)

                    if method == 'fourier':
                        self.ax_log.set_xlabel("log(Frequency)")
                        self.ax_log.set_ylabel("log(Power)")
                    else:
                        self.ax_log.set_xlabel("log(1/s)")
                        self.ax_log.set_ylabel("log(N(s))")

                self.canvas_log.draw()

        # Always update preview images (lightweight)

        if frame is not None:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.lbl_frame.setPixmap(QPixmap.fromImage(qt_image).scaled(self.lbl_frame.size(), Qt.KeepAspectRatio))

        if edges is not None:
            edges = np.ascontiguousarray(edges)
            h, w = edges.shape
            bytes_per_line = w
            qt_image = QImage(edges.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            self.lbl_edges.setPixmap(QPixmap.fromImage(qt_image).scaled(self.lbl_edges.size(), Qt.KeepAspectRatio))

        # Update Statistics & Histogram (every 10 frames)
        if len(self.results_data) % 10 == 0:
            Ds = [r['D'] for r in self.results_data]
            self.update_stats(Ds)

    def update_stats(self, Ds):
        if not Ds:
            return

        # Histogram
        self.ax_hist.clear()
        self.ax_hist.hist(Ds, bins=20, color=ACCENT, edgecolor=BG_DARK, alpha=0.85)
        self.ax_hist.axvline(1.3, color=ERROR_RED, linestyle='--', label='Optimal Low (1.3)')
        self.ax_hist.axvline(1.5, color=ERROR_RED, linestyle='--', label='Optimal High (1.5)')
        self.ax_hist.set_title("Distribution of D values")
        self.ax_hist.legend(facecolor=BG_SURFACE, edgecolor=BORDER, labelcolor=TEXT_PRIMARY)
        self._style_figure(self.fig_hist, self.ax_hist)
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
                # Save publication-quality plots (full timeline for D(t))
                self._save_timeseries_full(os.path.join(folder, f"fractal_timeseries_{base}.png"))
                self._save_fig_publication(self.fig_log, os.path.join(folder, f"fractal_loglog_{base}.png"))

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
                # Add Moisy-specific summary fields if applicable
                if 'D_std' in df.columns:
                    summary["mean_D_std"] = float(df['D_std'].mean())
                    summary["threshold"] = float(df['threshold'].iloc[0])
                    summary["padded_size"] = int(df['padded_size'].iloc[0])
                    summary["scale_range"] = str(df['scale_range'].iloc[0])

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

    def _save_timeseries_full(self, path):
        """Save the D(t) plot showing the complete timeline, regardless of current pan/zoom."""
        if not self.results_data:
            self._save_fig_publication(self.fig_time, path)
            return
        timestamps = [r['timestamp'] for r in self.results_data]
        orig_xlim = self.ax_time.get_xlim()
        t_min, t_max = min(timestamps), max(timestamps)
        margin = max((t_max - t_min) * 0.02, 1)
        self.ax_time.set_xlim(t_min - margin, t_max + margin)
        self._save_fig_publication(self.fig_time, path)
        self.ax_time.set_xlim(orig_xlim)
        self.canvas_time.draw()

    def export_results(self):
        if not self.results_data:
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if path:
            df = pd.DataFrame(self.results_data)
            df.to_csv(path, index=False)
            # Save publication-quality plots alongside CSV
            base = os.path.splitext(path)[0]
            self._save_timeseries_full(f"{base}_timeseries.png")
            self._save_fig_publication(self.fig_log, f"{base}_loglog.png")
            if len(self.results_data) >= 5:
                self._save_fig_publication(self.fig_hist, f"{base}_histogram.png")

def _set_title_bar_color(window, color_hex):
    """Set Windows title bar color using DWM API (Windows 11+)."""
    try:
        hwnd = int(window.winId())
        # Convert hex color to COLORREF (BGR format)
        r = int(color_hex[1:3], 16)
        g = int(color_hex[3:5], 16)
        b = int(color_hex[5:7], 16)
        color = r | (g << 8) | (b << 16)
        # DWMWA_CAPTION_COLOR = 35
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 35, ctypes.byref(ctypes.c_int(color)), ctypes.sizeof(ctypes.c_int)
        )
        # DWMWA_TEXT_COLOR = 36
        text_r = int(TEXT_PRIMARY[1:3], 16)
        text_g = int(TEXT_PRIMARY[3:5], 16)
        text_b = int(TEXT_PRIMARY[5:7], 16)
        text_color = text_r | (text_g << 8) | (text_b << 16)
        ctypes.windll.dwmapi.DwmSetWindowAttribute(
            hwnd, 36, ctypes.byref(ctypes.c_int(text_color)), ctypes.sizeof(ctypes.c_int)
        )
    except Exception:
        pass  # Silently fail on non-Windows or older Windows

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    _set_title_bar_color(window, BG_DARK)
    window.show()
    sys.exit(app.exec_())
