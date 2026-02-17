import cv2
import numpy as np
from src.core import FractalAnalyzer

def test_methods():
    analyzer = FractalAnalyzer()
    
    # Create synthetic images
    # 1. Solid square (filled) - D should be around 2 for box count of filled, but 1 for edges?
    # Wait, my box count is on edges. 
    # For DBC, a filled square is a flat surface (D=2 top, but intensity surface is flat?)
    # Actually DBC on flat image -> constant height -> boxes ~ scale^-2 -> D=2
    
    img = np.zeros((512, 512), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (400, 400), 255, -1)
    
    # Test Fourier
    print("Testing Fourier Slope...")
    D, R2, _, _ = analyzer.fourier_slope(img)
    print(f"Fourier D: {D:.4f}, R2: {R2:.4f}")
    
    # Test DBC
    print("Testing DBC...")
    D, R2, _, _ = analyzer.differential_box_count(img)
    print(f"DBC D: {D:.4f}, R2: {R2:.4f}")
    
    # Test Edge + Box
    print("Testing Edge + Box...")
    edges = cv2.Canny(img, 100, 200)
    D, R2, _, _, reliable = analyzer.box_count(edges)
    print(f"Edge+Box D: {D:.4f}, R2: {R2:.4f}, Reliable: {reliable}")

if __name__ == "__main__":
    test_methods()
