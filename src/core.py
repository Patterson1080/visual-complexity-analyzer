import cv2
import numpy as np
import scipy.stats

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False


def _to_gpu(arr):
    return cp.asarray(arr) if GPU_AVAILABLE else arr


def _to_cpu(arr):
    return cp.asnumpy(arr) if GPU_AVAILABLE and isinstance(arr, cp.ndarray) else arr


class FractalAnalyzer:
    def __init__(self):
        self.use_gpu = GPU_AVAILABLE

    @property
    def xp(self):
        return cp if self.use_gpu and GPU_AVAILABLE else np

    def preprocess_frame(self, frame, method='canny', threshold_mode='auto', 
                         manual_thresholds=(100, 200), blur_kernel=(5, 5)):
        """
        Preprocesses a frame for fractal analysis.
        Returns a binary edge image.
        """
        if frame is None:
            return None
            
        if len(frame.shape) == 2:
            # Already grayscale
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if blur_kernel:
            gray = cv2.GaussianBlur(gray, blur_kernel, 0)
            
        if method == 'canny':
            if threshold_mode == 'auto':
                median = np.median(gray)
                lower = int(max(0, 0.66 * median))
                upper = int(min(255, 1.33 * median))
                edges = cv2.Canny(gray, lower, upper)
            else:
                edges = cv2.Canny(gray, manual_thresholds[0], manual_thresholds[1])
        elif method == 'sobel':
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobelx**2 + sobely**2)
            # Normalize and threshold
            magnitude = np.uint8(255 * magnitude / np.max(magnitude))
            _, edges = cv2.threshold(magnitude, manual_thresholds[0], 255, cv2.THRESH_BINARY)
        else:
             # Default fallback
             edges = cv2.Canny(gray, 100, 200)

        # Ensure binary (0 or 1) for box counting, though Canny gives 0/255
        # We'll treat > 0 as edge
        return edges

    def box_count(self, binary_image, r2_threshold=0.90):
        """
        Calculates Fractal Dimension using 2D Box-Counting method.
        Returns: D, R_squared, scales (log(1/s)), counts (log(N(s))), reliable (bool)
        """
        if binary_image is None or np.sum(binary_image) == 0:
            return 0.0, 0.0, [], [], False

        xp = self.xp

        # Ensure binary 0/1 (use uint8 to minimize memory)
        pixels = _to_gpu((binary_image > 0).astype(np.uint8))

        # Minimal dimension
        H, W = pixels.shape
        MinimalDim = min(H, W)

        scales = []
        counts = []

        box_size = 2
        while box_size <= MinimalDim // 2:
            pad_h = (box_size - (H % box_size)) % box_size
            pad_w = (box_size - (W % box_size)) % box_size

            if pad_h > 0 or pad_w > 0:
                padded = xp.pad(pixels, ((0, pad_h), (0, pad_w)), mode='constant')
            else:
                padded = pixels

            sh = padded.shape
            reshaped = padded.reshape(sh[0] // box_size, box_size, sh[1] // box_size, box_size)
            block_sums = reshaped.sum(axis=(1, 3))

            non_empty_blocks = int(xp.count_nonzero(block_sums))

            if non_empty_blocks > 0:
                scales.append(1.0 / box_size)
                counts.append(non_empty_blocks)

            box_size *= 2

        if len(scales) < 2:
            return 0.0, 0.0, [], [], False

        # linregress on CPU (small arrays)
        log_scales = np.log(scales)
        log_counts = np.log(counts)

        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(log_scales, log_counts)

        D = slope  # D is the slope of log(N) vs log(1/s)
        R_squared = r_value ** 2

        # Quality checks: R² threshold and expected range for binary edge images
        reliable = R_squared >= r2_threshold and 1.0 <= D <= 2.0

        # Clamp D to valid range for binary edge images
        D = float(np.clip(D, 1.0, 2.0))

        return D, R_squared, log_scales, log_counts, reliable

    def differential_box_count(self, grayscale_image):
        """
        Differential Box Counting (DBC) for grayscale images.
        """
        if grayscale_image is None:
            return 0.0, 0.0, [], []

        xp = self.xp
        H, W = grayscale_image.shape
        MinimalDim = min(H, W)

        pixels = _to_gpu(grayscale_image)

        scales = []
        counts = []

        box_size = 2
        while box_size <= MinimalDim // 4:
            pad_h = (box_size - (H % box_size)) % box_size
            pad_w = (box_size - (W % box_size)) % box_size

            if pad_h > 0 or pad_w > 0:
                padded = xp.pad(pixels, ((0, pad_h), (0, pad_w)), mode='edge')
            else:
                padded = pixels

            sh = padded.shape
            reshaped = padded.reshape(sh[0] // box_size, box_size, sh[1] // box_size, box_size)

            mins = reshaped.min(axis=(1, 3))
            maxs = reshaped.max(axis=(1, 3))

            rs = maxs - mins + 1
            N_s = int(xp.sum(rs))

            if N_s > 0:
                scales.append(1.0 / box_size)
                counts.append(N_s)

            box_size *= 2

        if len(scales) < 2:
            return 0.0, 0.0, [], []

        log_scales = np.log(scales)
        log_counts = np.log(counts)
        slope, _, r_value, _, _ = scipy.stats.linregress(log_scales, log_counts)
        
        return slope, r_value**2, log_scales, log_counts

    def fourier_slope(self, grayscale_image):
        """
        Fourier Power Spectrum Slope (1/f analysis).
        D = (8 - beta) / 2
        """
        if grayscale_image is None:
            return 0.0, 0.0, [], []

        xp = self.xp
        img_gpu = _to_gpu(grayscale_image.astype(np.float64))

        # FFT2
        f = xp.fft.fft2(img_gpu)
        fshift = xp.fft.fftshift(f)

        # Radial Profile
        h, w = grayscale_image.shape
        center = (h // 2, w // 2)
        y, x = xp.ogrid[:h, :w]
        r = xp.sqrt((x - center[1])**2 + (y - center[0])**2)

        r_int = r.astype(int)

        power_spectrum = xp.abs(fshift)**2

        # Radial average
        tbin = xp.bincount(r_int.ravel(), power_spectrum.ravel())
        nr = xp.bincount(r_int.ravel())
        radial_profile = _to_cpu(tbin / xp.maximum(nr, 1))

        # Back to CPU for linregress
        max_r = min(h, w) // 2
        freqs = np.arange(1, max_r)
        powers = radial_profile[1:max_r]

        valid = (powers > 0)
        log_freqs = np.log(freqs[valid])
        log_powers = np.log(powers[valid])
        
        if len(log_freqs) < 5:
            return 0.0, 0.0, [], []
            
        # Fit P(f) proportional to f^(-beta)
        # log(P) = -beta * log(f) + C
        slope, _, r_value, _, _ = scipy.stats.linregress(log_freqs, log_powers)
        
        beta = -slope
        # D = (8 - beta) / 2 for 2D surfaces, approximation
        # But commonly D = (6 + beta)/2 ? Or D = (8 - beta)/2 ?
        # Prompt says: D = (8 - β) / 2
        D = (8 - beta) / 2
        
        return D, r_value**2, log_freqs, log_powers

    def generate_sierpinski_triangle(self, size=1024, n_points=500_000):
        """
        Generates a Sierpinski triangle via the chaos-game algorithm.

        The chaos game places individual pixels rather than filling polygons,
        avoiding rasterization aliasing and producing a result whose fractal
        dimension closely matches the theoretical log(3)/log(2) ≈ 1.5850.

        Args:
            size: Image dimension in pixels (square image).
            n_points: Number of chaos-game iterations. More points give a
                      denser, more accurate fractal.
        """
        image = np.zeros((size, size), dtype=np.uint8)

        # Vertices of an equilateral-ish triangle with a small margin
        margin = size // 20
        vertices = np.array([
            [size // 2, margin],              # top
            [margin, size - margin],           # bottom-left
            [size - margin, size - margin],    # bottom-right
        ], dtype=np.float64)

        # Start at a random vertex
        rng = np.random.default_rng(42)
        point = vertices[0].copy()

        # Pre-generate all random vertex choices at once
        choices = rng.integers(0, 3, size=n_points)

        # Run the chaos game: move halfway toward a randomly chosen vertex
        for idx in choices:
            point = (point + vertices[idx]) / 2.0
            px, py = int(point[0]), int(point[1])
            if 0 <= px < size and 0 <= py < size:
                image[py, px] = 255

        return image

    def generate_square(self, size=512, filled=False):
        """Generates a square for validation."""
        image = np.zeros((size, size), dtype=np.uint8)
        pad = size // 4
        if filled:
            cv2.rectangle(image, (pad, pad), (size-pad, size-pad), 255, -1)
        else:
            cv2.rectangle(image, (pad, pad), (size-pad, size-pad), 255, 1)
        return image
