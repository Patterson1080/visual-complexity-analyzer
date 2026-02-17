
# Validating Fractal Logic
import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from core import FractalAnalyzer

def main():
    analyzer = FractalAnalyzer()
    
    print("--- Validation Tests ---")
    
    # Test 1: Filled Square (should be D=1 for edges)
    # Why D=1 for edges? A filled square's boundary is just lines. 1D object.
    print("\nTest 1: Filled Square (Edge D approx 1.0)")
    square = analyzer.generate_square(filled=True)
    edges = analyzer.preprocess_frame(square, method='canny', threshold_mode='manual', manual_thresholds=(50, 150), blur_kernel=None)
    D, R2, _, _, reliable = analyzer.box_count(edges)
    print(f"Result: D={D:.4f}, R²={R2:.4f}, Reliable={reliable}")
    
    # Test 2: Sierpinski Triangle (Theoretical D approx 1.585)
    print("\nTest 2: Sierpinski Triangle (D approx 1.585)")
    sierpinski = analyzer.generate_sierpinski_triangle()
    # Note: If we edge detect a filled Sierpinski, we get the boundaries.
    # The boundary of a Sierpinski triangle is also fractal?
    # Actually, standard Sierpinski gasket is D ~ 1.585.
    # If we feed the binary image DIRECTLY to box count (skipping edge detection), we treat 'filled' areas as structure.
    # But our pipeline is Image -> Edge -> Box Count.
    # Let's see what happens with Canny. Canny on a filled Sierpinski will detect the edges of the triangles.
    # The edges of a Sierpinski gasket also scale with D ~ 1.585?
    # Let's try passing the image directly as 'edges' to box_count to verify the box_count logic itself first.
    
    print("  a) Direct Box Count on Binary Pattern (No Edge Detect):")
    D_direct, R2_direct, _, _, reliable_direct = analyzer.box_count(sierpinski)
    print(f"     Result: D={D_direct:.4f}, R²={R2_direct:.4f}, Reliable={reliable_direct}")
    
    print("  b) With Canny Edge Detection:")
    edges_sierp = analyzer.preprocess_frame(sierpinski, method='canny', blur_kernel=None)
    D_edge, R2_edge, _, _, reliable_edge = analyzer.box_count(edges_sierp)
    print(f"     Result: D={D_edge:.4f}, R²={R2_edge:.4f}, Reliable={reliable_edge}")

if __name__ == "__main__":
    main()
