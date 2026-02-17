# Video Fractal Dimensionality Analyzer

A tool that measures the visual complexity of video frames using fractal dimension analysis. Originally designed for environmental psychology research, it can be used by anyone interested in quantifying how "complex" or "detailed" a visual scene is.

## What is Fractal Dimension?

Fractal dimension (D) is a number that describes how complex a pattern is. Think of it as a score for visual detail:

- **D close to 1.0** — Very simple, smooth, minimal detail (a blank wall, clear sky)
- **D around 1.3–1.5** — Moderate complexity, often perceived as the most visually pleasing range (natural landscapes, trees, coastlines)
- **D close to 2.0** — Extremely complex, chaotic, visually overwhelming (dense static, tangled wires)

Research in environmental psychology suggests that people tend to prefer scenes with D values in the **1.3–1.5 range**, which is common in natural environments.

## Getting Started

### Requirements

- Python 3.8+
- Dependencies: `pip install PyQt5 matplotlib pandas opencv-python numpy scipy`

### Running

```
python main.py
```

## How to Use

1. **Load Video** — Click to open a video file (.mp4, .avi, .mov, .mkv)
2. **Start Analysis** — Begins processing frames and calculating fractal dimension over time
3. **Stop** — Stops analysis early if needed
4. **Batch Process Folder** — Analyze all videos in a folder automatically. Results (CSV, plot images, JSON summaries) are saved next to each video file
5. **Export Results** — Save the analysis data as a CSV file

## Analysis Methods

The app offers three different ways to calculate fractal dimension. Each has strengths depending on what you're analyzing.

### Edge + Box Counting (Default)

**Best for:** Analyzing the complexity of edges and outlines in a scene.

**How it works:** First, the app detects edges in the frame (the outlines of objects). Then it overlays a grid of boxes at different sizes and counts how many boxes contain part of an edge. The relationship between box size and count gives the fractal dimension.

**When to use it:** When you care about the complexity of shapes and boundaries — for example, comparing the silhouette complexity of a city skyline vs. a mountain range.

**Settings that matter:**
- *Edge Method:* `canny` (precise, good default) or `sobel` (softer, captures gradients)
- *Threshold Mode:* `auto` (recommended) or `manual`
- *Blur Kernel Size:* Higher values smooth out noise before edge detection. `5` is a good default

### Differential Box Counting (DBC)

**Best for:** Overall texture and surface complexity — the most robust general-purpose method.

**How it works:** Instead of detecting edges first, DBC works directly on the grayscale image. It treats pixel brightness as a 3D surface (like a terrain map) and measures how complex that surface is at different scales.

**When to use it:** When you want to measure the overall visual complexity of a scene, including textures, gradients, and subtle detail — not just hard edges. This is typically the **most accurate and reliable** method for video analysis.

**Note:** Edge detection settings (edge method, threshold, blur) are disabled for DBC since it doesn't use edge detection.

### Fourier Slope

**Best for:** Repetitive patterns and texture-heavy images.

**How it works:** The image is converted into frequency components (like how audio can be broken into bass, mid, and treble). The relationship between frequency and power follows a pattern that reveals fractal dimension.

**When to use it:** Specialized use — works well for images dominated by texture or noise-like patterns (ocean surfaces, cloud formations). Less reliable for typical video with mixed content like people, objects, and backgrounds.

## Understanding the Output

### Real-time Analysis Tab

- **Original Frame** — The current video frame being analyzed
- **Edge Detection** — The detected edges (for Edge + Box Counting method)
- **Log-Log Plot** — Shows the mathematical relationship used to calculate D. Straighter lines = more reliable results. If marked `[UNRELIABLE]` in red, the R² fit is poor and the D value for that frame may not be meaningful
- **D(t) Plot** — Fractal dimension over time, showing how visual complexity changes throughout the video

### Summary Tab

- **Statistics Table** — Mean, median, standard deviation, min, max, and percentage of frames in the optimal 1.3–1.5 range
- **Histogram** — Distribution of all D values with red dashed lines marking the 1.3–1.5 optimal range

## Settings

| Setting | What it does |
|---------|-------------|
| Sampling Rate | Analyze every Nth frame. Set to `1` for every frame, `10` to skip 9 out of 10 frames (faster but less detailed) |
| Edge Method | `canny` = sharp edge detection, `sobel` = gradient-based (softer edges). Only applies to Edge + Box Counting |
| Threshold Mode | `auto` = automatically determines edge sensitivity, `manual` = uses fixed values |
| Blur Kernel Size | Smoothing applied before edge detection. Higher = less noise but less fine detail. Use odd numbers (1, 3, 5, 7...) |
| Analysis Method | Choose between Edge + Box Counting, DBC, or Fourier Slope (see above) |
