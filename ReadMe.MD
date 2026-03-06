# entropy_viz

A Python tool that visualizes image complexity as an adaptive quadtree overlay using compression-based entropy scoring, with applications in AI-generated image detection and manipulation forensics.

---

## Overview

entropy_viz recursively subdivides an image into quadrants and scores each region for complexity. High-complexity regions — dense texture, fine detail, irregular structure — subdivide into smaller cells. Low-complexity regions — flat color, blurred backgrounds, repeated patterns — remain as large blocks. The result is an overlay where cell size and color both encode information density.

The core insight driving the project: compression ratio is a better proxy for true complexity than Shannon entropy. A blurred background has varied pixel values (high Shannon entropy) but compresses easily (low Kolmogorov complexity). zlib correctly identifies it as simple. Shannon does not.

---

## Install

```bash
pip install pillow numpy
```

Requires Python 3.7+.

---

## Usage

### Single image

```bash
# Compression entropy, fixed depth
python main.py photo.jpg --method compression --max-depth 6

# Shannon entropy, fixed depth
python main.py photo.jpg --method shannon --max-depth 5

# Adaptive splitting — stops when complexity drops below threshold
python main.py photo.jpg --method compression --threshold 0.4

# Both stopping conditions active
python main.py photo.jpg --method compression --max-depth 6 --threshold 0.4

# Save to custom path, no legend
python main.py photo.jpg -o result.png --no-legend
```

### Batch processing

Run across a folder of images, producing overlay PNGs and a `features.csv`:

```bash
python batch.py --input my_images/ --output results/ --label real
```

To build a labeled dataset across multiple folders, use `--append` to accumulate into the same CSV:

```bash
python batch.py --input real_photos/  --output results/ --label real
python batch.py --input ai_images/    --output results/ --label ai          --append
python batch.py --input photoshopped/ --output results/ --label photoshopped --append
```

To extract features only without rendering an overlay image per input:

```bash
python batch.py --input my_images/ --output results/ --label real --no-overlay
```

### RealVsFake dataset (wish096)

```bash
python3 batch.py --input "/home/penny/.cache/kagglehub/datasets/wish096/realvsfake-81k-by-wish/versions/3/RealVsFake/RealVsFake/Real" --output results/ --label real --no-overlay --workers 8

python3 batch.py --input "/home/penny/.cache/kagglehub/datasets/wish096/realvsfake-81k-by-wish/versions/3/RealVsFake/RealVsFake/Fake" --output results/ --label ai --append --no-overlay --workers 8
```

### Scatter plot

Visualize how well complexity features separate image classes:

```bash
# Default: mean_complexity vs std_complexity
python scatter.py results/features.csv

# Specify axes
python scatter.py results/features.csv --x mean_boundary_delta --y std_complexity

# Generate plots for all meaningful feature pairs (pruned)
python scatter.py results/features.csv --auto
```

---

## How It Works

**Scoring methods**

Two complexity scorers are available, both returning a normalized float in `[0, 1]`:

- **Shannon entropy** — measures pixel value distribution. Fast but blind to spatial repetition.
- **Compression ratio** — compresses raw pixel bytes with zlib and measures how much they shrink. Slower but more principled — it approximates Kolmogorov complexity from above.

Both scorers accept an optional opacity mask and only score opaque pixels, so transparent regions don't corrupt the complexity measurement of boundary nodes.

**Quadtree construction**

The image is recursively split into four quadrants. Splitting stops when any of the following are met:

- `max_depth` is reached (fixed mode)
- Region complexity falls below `threshold` (adaptive mode) — evaluated on opaque pixels only
- Region becomes smaller than `min_size` pixels
- Region is >= 95% transparent (background guard)

**Background masking**

For RGBA images, each node is tagged with `background_ratio` (fraction of transparent pixels). Nodes above `BG_THRESHOLD` (0.95) are excluded from complexity statistics and shown as neutral grey in the overlay.

**Visualization**

Subject nodes are colored by complexity score: blue (low) → green → yellow → red (high). Background nodes appear as neutral grey. The output PNG preserves the original alpha channel so transparent regions remain transparent.

**Feature extraction**

Each quadtree is distilled into a feature vector and written to `features.csv`:

| Feature | Description |
|---------|-------------|
| `mean_complexity` | Average complexity across subject leaf nodes |
| `std_complexity` | Variance — low in AI images which tend toward uniform complexity |
| `min_complexity` | Floor of complexity |
| `max_complexity` | Ceiling of complexity |
| `complexity_range` | max − min |
| `mean_leaf_area` | Average subject leaf area in pixels |
| `std_leaf_area` | Variance in leaf area |
| `leaf_count` | Total subject leaf count |
| `mean_boundary_delta` | Average complexity jump between parent and child nodes |
| `max_boundary_delta` | Maximum boundary jump — manipulation signal |
| `mean_depth` | Average depth of subject leaf nodes |
| `std_depth` | Variance in depth |

---

## Files

| File | Purpose |
|------|---------|
| `complexity.py` | Shannon entropy and compression ratio scorers, with opacity mask support |
| `quadtree.py` | `QuadNode` dataclass and `QuadTree` builder with background masking |
| `visualizer.py` | Overlay rendering, RGBA preservation, and colorbar legend |
| `features.py` | Feature extraction and CSV export |
| `scatter.py` | Scatter plot visualization of extracted features |
| `main.py` | Single image CLI entry point |
| `batch.py` | Batch processing, feature extraction, and labeled dataset pipeline |

---

## Options

**main.py**

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `shannon` | `shannon` or `compression` |
| `--max-depth` | `6` | Max split depth. `0` = adaptive only |
| `--threshold` | off | Stop splitting below this complexity score |
| `--min-size` | `8` | Minimum region side length (pixels) |
| `--alpha` | `120` | Overlay opacity (0–255) |
| `--no-borders` | off | Hide quadrant border lines |
| `--no-legend` | off | Omit colorbar legend |

**batch.py** (includes all main.py options plus):

| Flag | Default | Description |
|------|---------|-------------|
| `--label` | none | Label for all images in this batch (`real`, `ai`, `photoshopped`) |
| `--append` | off | Append to existing `features.csv` instead of overwriting |
| `--no-overlay` | off | Skip overlay rendering, extract features only |

**scatter.py**

| Flag | Default | Description |
|------|---------|-------------|
| `--x` | `mean_complexity` | Feature for the x axis |
| `--y` | `std_complexity` | Feature for the y axis |
| `--output` | `<csv>_scatter.png` | Output image path |
| `--auto` | off | Generate scatter plots for all feature pair combinations |

---

## Proof of Concept — AI Detection

The compression quadtree produces meaningfully different signatures for real photographs and AI-generated images. Initial results on the RealVsFake dataset (81k real vs AI-generated images) show visible separation in feature space using only `mean_complexity` and `std_complexity`:

![Proof of concept comparison](proof_of_concept.png)

Real images cluster toward lower mean complexity and higher variance — natural photographs have regions of simple and complex content. AI-generated images cluster toward higher mean complexity and lower variance — generators produce uniformly high complexity with little internal differentiation.

**What the quadtree reveals on individual images:**

- **Real photograph** — smooth complexity gradient. Face scores moderate (green), hair shows genuine internal variation. No hard discontinuities.
- **AI-generated** — hair and skin score uniformly high (red/orange) with little internal variation. The generator hallucinates plausible texture everywhere.
- **Photoshopped** — recolored regions are flattened in complexity. A hard discontinuity appears at the manipulation boundary.

These signatures emerge from no training data and no learned model — purely from information-theoretic properties of the image content.

---

## Known Limitations

- **Shannon entropy** produces a noisy grid at high depth due to statistical artifacts in small blocks. Use `--method compression` or raise `--min-size` to mitigate.
- **Compression thresholds** are not on the same scale as Shannon thresholds. Start around `0.4` for compression adaptive splitting.
- **Boundary nodes** that straddle the subject/background edge are scored on their opaque pixels only, which can produce elevated complexity scores if the subject content in that cell is detailed.
- **RealVsFake** images are face-centric — results may not generalize to non-portrait subjects without further validation.

---

## Future Plans

**Near term**
- ✅ Feature extraction: distill each quadtree into a feature vector exportable as CSV
- ✅ Scatter plot visualization of feature separation across labeled datasets
- Lightweight classifier: logistic regression on extracted features — no neural networks, architecture-agnostic by design

**Medium term**
- Merge-step delta scoring: measure the information gained at each split boundary — this is where manipulation boundaries show up most clearly
- Expand dataset: test against FFHQ (real) vs StyleGAN / Stable Diffusion outputs

**Longer term**
- Video pipeline: extract frames, build a complexity tensor across time, measure temporal stability of face-region complexity
- Audio-visual correlation: score whether facial complexity changes correlate with audio energy — a signal that breaks in dubbed or voice-synthesized deepfakes
- Benchmark against neural network deepfake detectors on unseen generative architectures — the architecture-agnostic approach should generalize where trained detectors fail

---

## Related Work

**Synthbuster** (2024) is the closest non-neural approach in the literature. It applies a cross-difference filter to highlight periodic artifacts in synthetic images, then extracts spectral features in the frequency domain and classifies them using gradient boosting. It targets GAN-specific upsampling artifacts — a different signal and a different domain from complexity-based analysis. Unlike Synthbuster, this project makes no assumptions about the generative architecture.

General frequency-domain analysis has shown that GAN-generated images exhibit periodic spectral artifacts caused by upsampling operations. Compression-based complexity is architecture-agnostic by design and does not rely on these artifacts, which may not be present in diffusion-model outputs.

---

## Background

The theoretical grounding for this project is Kolmogorov complexity — the length of the shortest program that produces a given output. True Kolmogorov complexity is uncomputable (proven via reduction to the halting problem), but compression ratio provides a computable upper bound. The project explores how far this approximation can take a practical detector.

The spatial distribution of complexity — not just a single score — is where the signal lives. A deepfake has a specific quadtree signature: a low-variance complexity island (the generated face) surrounded by a high-delta boundary ring, embedded in naturally varying terrain. That pattern is rare in unmanipulated photographs and detectable without any knowledge of the generative model that produced it.