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

```bash
# Compression entropy, fixed depth
python main.py photo.jpg --method compression --max-depth 6

# Shannon entropy, fixed depth
python main.py photo.jpg --method shannon --max-depth 5

# Adaptive splitting — stops when complexity drops below threshold
python main.py photo.jpg --method compression --threshold 0.4

# Both stopping conditions active
python main.py photo.jpg --method compression --max-depth 6 --threshold 0.4

# Batch process a folder
python batch.py --input my_images/ --output results/ --method compression
```

---

## How It Works

**Scoring methods**

Two complexity scorers are available, both returning a normalized float in `[0, 1]`:

- **Shannon entropy** — measures pixel value distribution. Fast but blind to spatial repetition..
- **Compression ratio** — compresses raw pixel bytes with zlib and measures how much they shrink. Slower but more principled — it approximates Kolmogorov complexity from above.

Both scorers accept an optional opacity mask and only score opaque pixels, so transparent regions don't corrupt the complexity measurement of boundary nodes.

**Quadtree construction**

The image is recursively split into four quadrants. Splitting stops when any of the following are met:

- `max_depth` is reached (fixed mode)
- Region complexity falls below `threshold` (adaptive mode) — evaluated on opaque pixels only
- Region becomes smaller than `min_size` pixels
- Region is >= 95% transparent (performance guard — no point subdividing empty space)

**Background masking**

For RGBA images, each node is tagged with `background_ratio` (fraction of transparent pixels). Nodes above `bg_threshold` (default 0.5) are excluded from complexity statistics and shown as neutral grey in the overlay rather than a heatmap color.

**Visualization**

Subject nodes are colored by complexity score: blue (low) → green → yellow → red (high). Background nodes appear as neutral grey. The output PNG preserves the original alpha channel so transparent regions remain transparent.

---

## Files

| File | Purpose |
|------|---------|
| `complexity.py` | Shannon entropy and compression ratio scorers, with opacity mask support |
| `quadtree.py` | `QuadNode` dataclass and `QuadTree` builder with background masking |
| `visualizer.py` | Overlay rendering, RGBA preservation, and colorbar legend |
| `main.py` | CLI entry point |
| `batch.py` | Batch processing across a folder of images |

---

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--method` | `shannon` | `shannon` or `compression` |
| `--max-depth` | `6` | Max split depth. `0` = adaptive only |
| `--threshold` | off | Stop splitting below this complexity score |
| `--min-size` | `8` | Minimum region side length (pixels) |
| `--bg-threshold` | `0.5` | Nodes with this fraction of transparent pixels are excluded from stats and shown as grey |
| `--alpha` | `120` | Overlay opacity (0–255) |
| `--no-borders` | off | Hide quadrant border lines |
| `--no-legend` | off | Omit colorbar legend |

---

## Proof of Concept — AI Detection

The compression quadtree produces meaningfully different signatures for real photographs, AI-generated images, and photoshopped images. The comparison below uses three portraits of models with dyed hair — one real photograph, one AI-generated, and one photograph with the hair color changed in Photoshop.

![Proof of concept comparison](proof_of_concept.png)

**What the quadtree reveals:**

- **Real photograph** — smooth complexity gradient across the subject. Face scores moderate (green), hair shows genuine internal variation with highlights and strand direction producing locally different scores. No hard discontinuities.

- **AI-generated** — hair region scores uniformly high (red/orange) with little internal variation. The generator hallucinates plausible texture everywhere, producing complexity that is high but suspiciously homogeneous.

- **Photoshopped hair** — the recolored hair is flattened in complexity relative to the face. A hard complexity discontinuity appears at the manipulation boundary — the compressor sees the recolored region as more repetitive than surrounding original content.

These signatures emerge from no training data and no learned model — purely from information-theoretic properties of the image content.

---

## Known Limitations

- **Shannon entropy** produces a noisy grid at high depth (blocking artifact) because small blocks score erratically. Use `--method compression` or raise `--min-size` to mitigate.
- **Compression thresholds** are not on the same scale as Shannon thresholds. Start around `0.4` for compression adaptive splitting.
- **Boundary nodes** that straddle the subject/background edge are scored on their opaque pixels only, which can produce high complexity scores if the subject content in that cell is detailed.

---

## Future Plans

**Near term**
- Feature extraction: distill each quadtree into a small feature vector (mean complexity, internal variance, boundary delta sharpness) exportable as CSV
- Comparison mode: diff two images' complexity profiles side by side with a delta visualization

**Medium term**
- Labeled dataset pipeline: run the scorer across folders of known-real and known-AI images, output per-image feature CSVs, visualize population separation
- Lightweight classifier: logistic regression on extracted features — no neural networks, architecture-agnostic by design
- Merge-step delta scoring: measure the information gained at each split boundary — this is where manipulation boundaries show up most clearly

**Longer term**
- Video pipeline: extract frames, build a complexity tensor across time, measure temporal stability of face-region complexity frame-to-frame
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