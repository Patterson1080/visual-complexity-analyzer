import sys
import os
import time
from PyQt5.QtWidgets import QApplication
from src.gui import MainWindow

def test_crash():
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Simulate loading a video
    # We need a dummy video file. Let's create one using OpenCV if it doesn't exist.
    video_path = os.path.abspath("test_video.mp4")
    
    import cv2
    import numpy as np
    
    if not os.path.exists(video_path):
        print("Creating dummy video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))
        for _ in range(30):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
        out.release()
        
    print(f"Loading video: {video_path}")
    window.current_video_path = video_path
    
    # Check settings
    print("Starting analysis...")
    window.start_analysis()
    
    # Keep app running for a few seconds to allow thread to start and potentially crash
    start_time = time.time()
    while time.time() - start_time < 5:
        app.processEvents()
        time.sleep(0.1)
        
    print("Finished waiting. If no crash output above, it might be fixed.")
    if window.analysis_thread and window.analysis_thread.isRunning():
        print("Thread is still running.")
        window.stop_analysis()
        window.analysis_thread.wait()
    else:
        print("Thread finished or crashed.")

if __name__ == "__main__":
    test_crash()
