# Video Fractal Dimensionality Analyzer

A tool that measures the visual complexity of video frames using fractal dimension analysis. Originally designed for environmental psychology research, it can be used by anyone interested in quantifying how "complex" or "detailed" a visual scene is.

## What is Fractal Dimension?

Fractal dimension (D) is a number that describes how complex a pattern is. Think of it as a score for visual detail:

- **D close to 1.0** — Very simple, smooth, minimal detail (a blank wall, clear sky)
- **D around 1.3–1.5** — Moderate complexity, often perceived as the most visually pleasing range (natural landscapes, trees, coastlines)
- **D close to 2.0** — Extremely complex, chaotic, visually overwhelming (dense static, tangled wires)

Research in environmental psychology suggests that people tend to prefer scenes with D values in the **1.3–1.5 range**, which is common in natural environments.

## Getting Started

### Installation

```bash
git clone https://github.com/Patterson1080/visual-complexity-analyzer.git
cd visual-complexity-analyzer
pip install -r requirements.txt
python main.py
```

**Requirements:** Python 3.8+

## How to Use

1. **Load Video** — Click to open a video file (.mp4, .avi, .mov, .mkv). The Clip Range fields automatically populate with the video's duration
2. **Set Clip Range** *(optional)* — Use the `HH:MM:SS → HH:MM:SS` fields to restrict analysis to a specific portion of the video. Useful for skipping black leaders/endings or focusing on a particular scene. Setting the end time to `00:00:00` analyzes to the end of the video
3. **Start Analysis** — Begins processing frames and calculating fractal dimension over time. Progress bar reflects only the selected clip range
4. **Stop** — Stops analysis early if needed
5. **Batch Process Folder** — Analyze all videos in a folder automatically. Results (CSV, plot images, JSON summaries) are saved next to each video file
6. **Export Results** — Save the analysis data as a CSV file. The exported D(t) timeseries plot always shows the **complete analyzed timeline**, regardless of the current pan/zoom view

## Analysis Methods

The app offers four different ways to calculate fractal dimension. Each has strengths depending on what you're analyzing.

### Moisy Threshold + Box Counting (Default)

**Best for:** Reproducing results from published research using Frédéric Moisy's MATLAB `boxcount` function.

**How it works:** Converts each frame to grayscale, applies a simple brightness threshold to produce a binary (black/white) image, then runs a coarsening box-count: the image is repeatedly halved by combining 2×2 blocks via logical OR, building up counts at each power-of-2 scale. Fractal dimension is computed as the average of local log-log slopes over a configurable mid-range scale window — matching Moisy's MATLAB algorithm exactly.

**When to use it:** When you need results that are directly comparable to collaborators using Moisy's MATLAB boxcount, or when you want a straightforward threshold-based measurement of binary image complexity. Validated against MATLAB output to within ±0.001 per frame.

**Settings that matter:**
- *Binarization Threshold:* Fraction of max brightness (0–1) above which pixels become foreground. Default `0.25` matches the published method
- *Scale Range:* MATLAB-indexed range of local slopes to average. Default `4–8` uses mid-range scales, matching the standard published approach

**Note:** This method intentionally uses no edge detection and no blurring. Typical D values on natural video are ~1.37–1.53, lower than the Edge + Box Counting method because dense binary blobs produce different box-count behavior than thin Canny edges.

### Edge + Box Counting

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

### Why do methods give different D values?

The same video can yield different D values depending on the method — this is expected, not an error. Each method measures a different geometric property:

| Factor | Edge + Box Counting | Moisy Threshold |
|--------|--------------------|--------------------|
| Preprocessing | Canny edges (thin, sparse) | Brightness threshold (dense blobs) |
| Typical D on natural video | ~1.6–1.8 | ~1.37–1.53 |
| FD computation | Global linear regression (all scales) | Local slopes averaged over mid-range |
| MATLAB comparable | No | Yes (Moisy boxcount) |

## Understanding the Output

### Real-time Analysis Tab

- **Original Frame** — The current video frame being analyzed
- **Processed Frame** — The detected edges (Edge + Box Counting) or binarized image (Moisy method)
- **Log-Log Plot** — Shows the mathematical relationship used to calculate D. For the Moisy method, gold markers highlight the scale range used to compute D, and the title shows D ± std. For other methods, if marked `[UNRELIABLE]` in red the R² fit is poor and the D value may not be meaningful
- **D(t) Plot** — Fractal dimension over time. The interactive view uses a sliding 30-second window; the exported PNG always shows the full timeline

### Summary Tab

- **Statistics Table** — Mean, median, standard deviation, min, max, and percentage of frames in the optimal 1.3–1.5 range
- **Histogram** — Distribution of all D values with red dashed lines marking the 1.3–1.5 optimal range

## Settings

| Setting | What it does |
|---------|-------------|
| Clip Range | `HH:MM:SS → HH:MM:SS` start and end times. Auto-filled from video duration on load. End `00:00:00` = analyze to end of video |
| Sampling Rate | Analyze every Nth frame. Set to `1` for every frame, `10` to skip 9 out of 10 frames (faster but less detailed) |
| Analysis Method | Choose between Moisy Threshold + Box Counting (default), Edge + Box Counting, DBC, or Fourier Slope |
| Binarization Threshold | *(Moisy only)* Brightness cutoff (0–1) for grayscale→binary conversion. Default `0.25` matches the published method |
| Scale Range | *(Moisy only)* MATLAB-indexed range of local slopes to average. Default `4–8`. Wider range = smoother estimate; narrower = more sensitive to a specific scale |
| Edge Method | `canny` = sharp edge detection, `sobel` = gradient-based (softer edges). Only applies to Edge + Box Counting |
| Threshold Mode | `auto` = automatically determines edge sensitivity, `manual` = uses fixed values. Only applies to Edge + Box Counting |
| Blur Kernel Size | Smoothing applied before edge detection. Higher = less noise but less fine detail. Use odd numbers (1, 3, 5, 7...). Only applies to Edge + Box Counting |
