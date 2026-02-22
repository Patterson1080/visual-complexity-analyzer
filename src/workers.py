import math

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
import time
from src.core import FractalAnalyzer

class AnalysisThread(QThread):
    progress_updated = pyqtSignal(int, int) # current_frame, total_frames
    frame_processed = pyqtSignal(dict) # result data dictionary
    analysis_finished = pyqtSignal()
    
    def __init__(self, video_path, settings=None):
        super().__init__()
        self.video_path = video_path
        self.settings = settings if settings else {}
        self._is_running = True
        self.analyzer = FractalAnalyzer()

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {self.video_path}")
                return
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Sampling rate
            sampling_rate = self.settings.get('sampling_rate', 1)

            # Clip range (seconds → frames)
            clip_start_sec = self.settings.get('clip_start_sec', 0)
            clip_end_sec   = self.settings.get('clip_end_sec', 0)

            start_frame = int(clip_start_sec * fps) if fps > 0 else 0
            start_frame = max(0, min(start_frame, total_frames - 1))

            if clip_end_sec > 0:
                end_frame = int(clip_end_sec * fps) if fps > 0 else total_frames
                end_frame = max(start_frame + 1, min(end_frame, total_frames))
            else:
                end_frame = total_frames  # 00:00:00 end = full video

            # Seek to start
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frame_idx = start_frame
            clip_total = end_frame - start_frame  # for progress bar

            while self._is_running:
                if frame_idx >= end_frame:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                if (frame_idx - start_frame) % sampling_rate == 0:
                    try:
                        # Process frame
                        # Process frame based on method
                        analysis_type = self.settings.get('analysis_type', 'box_counting')
                        
                        D = 0.0
                        R2 = 0.0
                        log_scales = []
                        log_counts = []
                        edges = None
                        
                        # Check if we need grayscale first
                        if len(frame.shape) == 3:
                             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                             gray = frame
                        
                        reliable = True

                        if analysis_type == 'moisy_boxcount':
                            moisy_thresh = self.settings.get('moisy_threshold', 0.25)
                            scale_range = self.settings.get('scale_range', (4, 8))
                            D, D_std, n, r, df, bw = self.analyzer.analyze_frame_moisy(
                                frame, threshold=moisy_thresh, scale_range=scale_range)
                            # Padded size for metadata
                            padded_p = math.ceil(math.log2(max(gray.shape)))
                            padded_size = 2 ** padded_p
                            # Use log(r) and log(n) for the log-log plot
                            log_scales = np.log(r.astype(float)) if len(r) > 0 else []
                            log_counts = np.log(n.astype(float)) if len(n) > 0 else []
                            R2 = 0.0  # Not applicable for local-slope method
                            reliable = True
                            # Store binarized image as preview (uint8 for display)
                            edges = (bw.astype(np.uint8) * 255)

                        elif analysis_type == 'box_counting':
                            method = self.settings.get('edge_method', 'canny')
                            threshold_mode = self.settings.get('threshold_mode', 'auto')
                            manual_thresholds = self.settings.get('manual_thresholds', (100, 200))
                            blur_kernel_size = self.settings.get('blur_kernel_size', 5)
                            blur_kernel = (blur_kernel_size, blur_kernel_size) if blur_kernel_size > 0 else None

                            edges = self.analyzer.preprocess_frame(frame, method, threshold_mode, manual_thresholds, blur_kernel)
                            D, R2, log_scales, log_counts, reliable = self.analyzer.box_count(edges)

                        elif analysis_type == 'dbc':
                            # Differential Box Counting (uses grayscale)
                            D, R2, log_scales, log_counts = self.analyzer.differential_box_count(gray)
                            edges = gray # Show grayscale in preview instead of edges?
                            
                        elif analysis_type == 'fourier':
                            # Fourier Slope
                            D, R2, log_scales, log_counts = self.analyzer.fourier_slope(gray)
                            edges = gray # Show grayscale
                        
                        result = {
                            'frame_idx': frame_idx,
                            'timestamp': frame_idx / fps if fps > 0 else 0,
                            'D': D,
                            'R2': R2,
                            'reliable': reliable,
                            'scales': log_scales,
                            'counts': log_counts,
                            'edge_pixels': cv2.countNonZero(edges) if (edges is not None and analysis_type == 'box_counting') else 0,
                            'frame': frame,
                            'edges': edges,
                            'method': analysis_type
                        }

                        # Moisy-specific fields
                        if analysis_type == 'moisy_boxcount':
                            result['D_std'] = D_std
                            result['threshold'] = moisy_thresh
                            result['padded_size'] = padded_size
                            result['scale_range'] = f"{scale_range[0]}-{scale_range[1]}"
                            result['df'] = df  # local slopes for log-log highlight
                        
                        self.frame_processed.emit(result)
                        
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"Error processing frame {frame_idx}: {e}")
                
                self.progress_updated.emit(frame_idx - start_frame, clip_total)
                frame_idx += 1
                
            cap.release()
            self.analysis_finished.emit()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Critical error in AnalysisThread: {e}")

    def stop(self):
        self._is_running = False
