# Entropy Quadtree

A tool for visualizing and classifying image complexity using adaptive quadtrees.
Originally designed to distinguish real photographs from AI-generated and photoshopped images,
though the feature set is general-purpose for any image complexity analysis task.

The core idea: recursively subdivide an image into quadrants, score each region's complexity,
and use the resulting tree structure — not just the scores themselves — as a rich feature vector
for classification.

---

## How it works

A **quadtree** is built over each image by recursively splitting regions into four quadrants
until a minimum leaf size is reached. Each region is scored for complexity using one of three
methods:

- **Shannon entropy** — pixel value distribution. Fast and works at any leaf size.
- **Compression ratio** — zlib compressibility. Captures repetition that entropy misses (reliable at `--leaf_size 16`).
- **Variance** — normalised pixel variance. Fastest; a good proxy for compression.

Background regions (identified via alpha channel) are excluded from scoring and statistics.
The resulting tree is then used to extract a feature vector for each image.

### Example output

![Example overlays showing AI-generated, photoshopped, and photographed images](results/examples/proof_of_concept.png)

Each pair shows the original image alongside its complexity overlay. Several patterns are
immediately visible:

- **AI-generated** (top left): the hair is saturated red — extremely high complexity from the
  chaotic strand-level detail that diffusion models generate. The face and clothing sit in
  mid-range green/yellow.
- **Photoshopped** (bottom left): the face is deep blue — unusually *low* complexity for skin,
  a signature of retouching that smooths texture. The hair is green/orange, much simpler than
  the AI example despite being similarly coloured.
- **Photographed** (bottom right): more spatially uniform green/blue throughout, with gentle
  variation. No extreme highs or lows — the natural complexity gradient of an unmodified photo.

---

## File overview

| File | Purpose |
|---|---|
| `src/main.py` | Single-image visualizer — renders a complexity overlay PNG |
| `src/batch.py` | Multi-folder batch processor — extracts features to CSV |
| `src/stream_batch.py` | Streams images directly from a Kaggle dataset without writing to disk |
| `src/classify.py` | Trains and evaluates classifiers on an extracted feature CSV |
| `src/merge_features.py` | Merges per-method feature CSVs into a wide CSV with cross-method features |
| `src/grid_importance.py` | Heatmap of which 16×16 spatial grid cells matter most for classification |
| `src/scatter.py` | Scatter plots of feature pairs across a labeled dataset |
| `src/tune_thresholds.py` | Sweeps the pruning threshold hyperparameter, finds the best value |
| `src/tune_plots.py` | Plots output from `tune_thresholds.py` |
| `src/depth_distribution.py` | Analyses tree depth across a dataset to calibrate grid depth |
| `src/quadtree.py` | Core quadtree builder (`QuadTree`, `QuadNode`) |
| `src/complexity.py` | Scoring functions: `shannon_entropy`, `compression_entropy`, `variance_complexity` |
| `src/features.py` | Feature extraction from a built tree (`extract_features`, CSV I/O) |
| `src/visualizer.py` | Overlay rendering and colormap |

---

## Installation

```bash
pip install pillow numpy scikit-learn matplotlib
```

For Kaggle streaming (`stream_batch.py`):

```bash
pip install kaggle requests
```

---

## Quickstart

### 1. Visualize a single image

```bash
python3 src/main.py photo.jpg
python3 src/main.py photo.jpg --method compression --leaf_size 16
python3 src/main.py photo.jpg --method shannon --legend --borders
python3 src/main.py photo.jpg -o result.png --alpha 160
```

Produces `photo_entropy.png` — the original image with a complexity heatmap overlaid.
Blue = low complexity, red = high complexity.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--method` | `shannon` | `shannon`, `compression`, or `variance` |
| `--leaf_size` | `4` | Target leaf side length in pixels. Smaller = finer detail |
| `--threshold` | off | Percentile pruning cutoff — collapses the least complex N% of leaves |
| `--alpha` | `120` | Overlay opacity (0–255) |
| `--borders` | off | Draw quadrant border lines |
| `--legend` | off | Append a colorbar legend below the image |
| `-o` | `<input>_entropy.png` | Output path |

---

### 2. Tune the pruning threshold

Before extracting features at scale, find the optimal `--threshold` value for your dataset and
method. The threshold controls what fraction of the least complex leaves are collapsed back into
their parent — trading resolution for noise reduction. `tune_thresholds.py` builds each image's
tree once and then applies different prune cutoffs in memory, so the sweep is nearly as fast as
a single batch run.

```bash
# Coarse sweep (0, 10, 20, ... 60) — good starting point
python3 src/tune_thresholds.py \
    --input real_photos ai_images \
    --labels real ai

# Recommended starting config for compression
python3 src/tune_thresholds.py \
    --input real_photos ai_images \
    --labels real ai \
    --method compression --leaf_size 16

# Fast pilot run before committing to the full dataset
python3 src/tune_thresholds.py \
    --input real_photos ai_images \
    --labels real ai \
    --max_images 200

# Refine around a known peak, appending to an existing results CSV
python3 src/tune_thresholds.py \
    --input real_photos ai_images \
    --labels real ai \
    --method compression \
    --thresholds 28 30 32 34 35 36 38 40 \
    --append
```

Then visualize the sweep:

```bash
# Default: reads results/tuning/{shannon,compression,variance}.csv automatically
python3 src/tune_plots.py

# Explicit input file
python3 src/tune_plots.py --input results/tuning/compression.csv
```

Writes `results/tuning/` → `accuracy_vs_thresholds.png`, `leaf_count_vs_thresholds.png`, `accuracy_vs_leaf_count.png`.
The script prints the optimal threshold and a ready-to-run `batch.py` command at the end.

---

### 3. Calibrate tree-grid depth

The spatial tree-grid feature (256 cells) is built by reading the quadtree at a fixed depth.
`depth_distribution.py` samples your dataset to verify every image reliably reaches that depth
before you commit to a full batch run.

```bash
# Default output: results/depth/{method}_spatial_map.png
python3 src/depth_distribution.py \
    --input real_photos ai_images \
    --labels real ai \
    --method compression \
    --max_images 500

# Run all three methods at once
python3 src/depth_distribution.py \
    --input real_photos ai_images \
    --labels real ai \
    --output results/depth/
```

---

### 4. Extract features from a dataset

#### From local folders

Use the threshold found in step 2 in the `--threshold` flag here.

```bash
# Single folder, unlabeled — auto-named results/features/shannon.csv
python3 src/batch.py --input my_images --labels unlabeled --output results/features/

# Multiple folders in one pass — builds a labeled dataset
python3 src/batch.py \
    --input real_photos ai_images \
    --labels real ai \
    --output results/features/

# Append a third class to an existing CSV
python3 src/batch.py \
    --input photoshopped \
    --labels photoshopped \
    --output results/features/shannon.csv \
    --append

# Compression method with tuned threshold (better signal, use leaf_size 16)
python3 src/batch.py \
    --input real_photos ai_images \
    --labels real ai \
    --output results/features/ \
    --method compression --leaf_size 16 --threshold 32

# Parallel processing
python3 src/batch.py \
    --input real_photos ai_images \
    --labels real ai \
    --output results/features/ \
    --workers 8
```

#### From Kaggle (no disk storage)

`stream_batch.py` downloads a Kaggle dataset zip into memory and processes images one by one
without ever writing them to disk. Useful for large datasets or environments with limited
storage. Supports multiple scoring methods in a single pass (one download, N output CSVs).

```bash
# Explore a dataset's folder structure before committing
python3 src/stream_batch.py \
    --dataset ciplab/real-and-fake-face-detection \
    --explore

# Single method
python3 src/stream_batch.py \
    --dataset ciplab/real-and-fake-face-detection \
    --classes training_fake training_real \
    --labels manipulated authentic \
    --label_detail gan_faceswap authentic_portrait \
    --prefix real_and_fake_face \
    --methods compression \
    --resize 256

# Two methods in one pass (one download, two output CSVs)
python3 src/stream_batch.py \
    --dataset ciplab/real-and-fake-face-detection \
    --classes training_fake training_real \
    --labels manipulated authentic \
    --label_detail gan_faceswap authentic_portrait \
    --prefix real_and_fake_face \
    --methods compression shannon \
    --resize 256 --workers 8

# Append a second Kaggle dataset to the same CSVs
python3 src/stream_batch.py \
    --dataset wish096/realvsfake-81k-by-wish \
    --classes Fake Real \
    --labels synthetic authentic \
    --label_detail gan_portrait authentic_portrait \
    --prefix RealVsFake/RealVsFake \
    --methods compression shannon \
    --resize 256 --max_per_dataset 2000 --workers 8 \
    --append
```

Produces a CSV with one row per image and 282 feature columns (26 scalar features + 256 spatial grid cells). Output files are named `{method}.csv` — e.g. `results/features/compression.csv`.

---

### 5. Merge features from multiple methods (optional)

Shannon entropy and compression ratio are partially independent signals — AI-generated regions
often have high Shannon entropy but low compression ratio because their noise is structurally
repetitive in ways zlib can exploit. `merge_features.py` combines per-method CSVs into a single
wide CSV and adds cross-method ratio/delta features that directly encode this divergence.

```bash
# Merge shannon + compression (most common case)
python3 src/merge_features.py \
    --inputs results/features/shannon.csv results/features/compression.csv \
    --methods shannon compression \
    --output results/features/merged_shannon_compression.csv

# Three-way merge
python3 src/merge_features.py \
    --inputs results/features/shannon.csv results/features/compression.csv results/features/variance.csv \
    --methods shannon compression variance \
    --output results/features/merged_all.csv
```

The output includes all per-method features (prefixed with the method name) plus cross-method
features like `cross_shannon_compression_mean_complexity_ratio` — the ratio of Shannon to
compression complexity, a strong AI-vs-real discriminator. Pass the merged CSV directly to
`classify.py`.

---

### 6. Train classifiers

```bash
# Full suite (Logistic Regression, Linear SVM, Random Forest, Gradient Boosting)
python3 src/classify.py results/features/compression.csv

# Fast mode (Logistic Regression + Random Forest only)
python3 src/classify.py results/features/compression.csv --fast

# Balance classes, show permutation importance
python3 src/classify.py results/features/compression.csv --balance --permutation-importance --fast

# Prune low-signal features before training
python3 src/classify.py results/features/compression.csv --prune-features 0.0 --fast
python3 src/classify.py results/features/compression.csv --prune-features 0.0005 --fast

# Classify on merged multi-method features
python3 src/classify.py results/features/merged_shannon_compression.csv --fast --balance

# Use only a subset of features
python3 src/classify.py results/features/compression.csv \
    --features mean_complexity std_complexity max_merge_delta mean_boundary_delta
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--fast` | off | Run only Logistic Regression and Random Forest |
| `--balance` | off | Undersample majority class to equalize class counts |
| `--test-size` | `0.2` | Fraction held out for testing |
| `--cv` | `5` | Cross-validation folds |
| `--features` | all | Subset of feature columns to use |
| `--permutation-importance` | off | Compute permutation importance for RF (slower, more reliable than impurity importance for correlated features) |
| `--prune-features THRESHOLD` | off | Fit a probe RF, drop features below `THRESHOLD` permutation importance, then run classifiers on the reduced set. `0.0` drops only harmful features; `0.0005` is more aggressive |
| `--output` | `results/classify/{csv_stem}.txt` | Path for the saved report |

Output includes accuracy, precision, recall, F1, confusion matrix, feature importances, and a summary ranking.

#### Note on WSL

If you're running on WSL and hit `OSError: [Errno 12] Cannot allocate memory`, this is a joblib
fork issue. The fix is already included in `classify.py` via `joblib.parallel_backend("threading")`.

---

### 7. Visualize spatial importance

`grid_importance.py` fits a Random Forest on a features CSV and renders the permutation
importance of all 256 `tree_grid_*` cells as a 16×16 heatmap — showing exactly which regions
of the image the classifier relies on most. Optionally overlays the heatmap on a reference image.

```bash
# Heatmap only
python3 src/grid_importance.py results/features/compression.csv

# Overlay on a reference image
python3 src/grid_importance.py results/features/compression.csv \
    --image real_00079.jpg

# Show pairwise class comparison (which cells separate real vs ai)
python3 src/grid_importance.py results/features/compression.csv \
    --compare real ai

# Custom output path
python3 src/grid_importance.py results/features/compression.csv \
    --output results/grid_importance/compression.png
```

Output: `results/grid_importance/{csv_stem}.png`

---

### 8. Scatter plots

```bash
# Default axes (mean_complexity vs std_complexity)
# Output: results/scatter/compression/scatter_compression.png
python3 src/scatter.py results/features/compression.csv

# Custom axes
python3 src/scatter.py results/features/compression.csv --x mean_boundary_delta --y max_merge_delta

# Generate all non-redundant feature pair plots
python3 src/scatter.py results/features/compression.csv --auto

# Skip highly correlated pairs (default |r| >= 0.9)
python3 src/scatter.py results/features/compression.csv --auto --corr-threshold 0.85
```

---

## Feature reference

Each image produces 282 features (26 scalar + 256 spatial).

### Scalar features (26)

| Feature | Description |
|---|---|
| `mean_complexity` | Average complexity across subject leaves |
| `std_complexity` | Variance of complexity — tends to be lower in AI images |
| `min_complexity` | Floor of the complexity distribution |
| `max_complexity` | Ceiling — almost always 1.0 |
| `complexity_range` | `max - min` — spread of the distribution |
| `mean_leaf_area` | Average subject leaf area in pixels |
| `std_leaf_area` | Variance in leaf area — high values indicate adaptive splitting |
| `leaf_count` | Total number of subject leaves |
| `mean_boundary_delta` | Average complexity difference between parent and child nodes |
| `max_boundary_delta` | Peak boundary jump — a manipulation signal |
| `mean_depth` | Average tree depth across subject leaves |
| `std_depth` | Variance in depth |
| `mean_merge_delta` | Average `std(children) / parent` across all splits — information gain per split |
| `max_merge_delta` | Peak merge delta — highlights manipulation boundaries |
| `std_merge_delta` | Variance of merge deltas — high in real images, low in AI |
| `mean_complexity_r/g/b` | Per-channel mean complexity across subject leaves |
| `complexity_delta_rg/rb/gb` | Inter-channel complexity divergence |

### Spatial features (256)

`tree_grid_000` through `tree_grid_255` — complexity values at each cell of a 16×16
equivalent spatial grid, sampled at tree depth 4 in Morton order (top-left to bottom-right).
Cells where the tree stopped early propagate their parent's complexity downward, so the vector
is always exactly 256 elements regardless of image resolution.

These spatial features capture *where* complexity is concentrated in the image. In practice,
several specific grid cells (particularly in the lower quadrants) rank among the most
informative features for AI vs real classification, reflecting the characteristic spatial
signatures of generative models. Use `grid_importance.py` to visualize which cells matter most
for your dataset.

---

## Typical workflow

```
real_photos/                 ai_images/
      │                          │
      └─ src/tune_thresholds.py ─┘   ← find optimal --threshold
                  │                    → results/tuning/{method}.csv
          src/tune_plots.py           ← visualize the sweep
                  │                    → results/tuning/*.png
           ┌──────┴──────────────────────┐
src/depth_distribution.py   src/batch.py --threshold <N>
                             src/stream_batch.py (Kaggle)
                                         │
                                         ↓  results/features/{method}.csv
                             src/merge_features.py (optional)
                                         │
                                         ↓  results/features/merged_*.csv
                  ┌──────────────────────┼──────────────────────┐
           src/classify.py         src/scatter.py     src/grid_importance.py
                  │                      │                      │
          results/classify/       results/scatter/   results/grid_importance/
```

A full run from scratch:

```bash
# 1. Pilot sweep to find the best pruning threshold (~200 images per class)
python3 src/tune_thresholds.py \
    --input data/real data/ai \
    --labels real ai \
    --method compression --leaf_size 16 \
    --max_images 200
python3 src/tune_plots.py

# 2. Verify tree depth is consistent across the dataset
python3 src/depth_distribution.py \
    --input data/real data/ai \
    --labels real ai \
    --method compression

# 3. Extract features at scale using the tuned threshold
python3 src/batch.py \
    --input data/real data/ai \
    --labels real ai \
    --output results/features/ \
    --method compression --threshold 32 --workers 8

# 4. Quick classification check
python3 src/classify.py results/features/compression.csv --fast --balance

# 5. Prune low-signal features and re-evaluate
python3 src/classify.py results/features/compression.csv \
    --fast --balance --prune-features 0.0 --permutation-importance

# 6. Visualize which spatial cells matter most
python3 src/grid_importance.py results/features/compression.csv

# 7. Visualize feature separation
python3 src/scatter.py results/features/compression.csv --auto
```

---

## Related work

This project sits at the intersection of image forensics and complexity analysis. Some relevant
prior work:

**Image forgery detection**
- Farid, H. (2009). *Image forgery detection*. IEEE Signal Processing Magazine. An early survey
  of pixel-level manipulation signals — many of the boundary delta and noise inconsistency ideas
  here are inspired by this line of work.
- Krawetz, N. *A Picture's Worth: Digital Image Analysis and Forensics* (Black Hat, 2007).
  Error level analysis (ELA), a precursor to compression-ratio approaches for detecting
  re-saved or composited regions.

**AI image detection**
- Wang, S. Y. et al. (2020). *CNN-generated images are surprisingly easy to spot... for now.*
  CVPR. Establishes that low-level frequency artifacts in GAN outputs are detectable; the
  compression-ratio method here captures related repetition artifacts without requiring a
  learned detector.
- Corvi, R. et al. (2023). *On the detection of synthetic images generated by diffusion models.*
  ICASSP. Notes that diffusion model outputs are harder to detect than GAN outputs by
  frequency-domain methods — motivating the spatial complexity map approach here.

**Quadtree image analysis**
- Samet, H. (1984). *The quadtree and related hierarchical data structures.* ACM Computing
  Surveys. Foundational reference for quadtree spatial decomposition.
- Hurtut, T. et al. (2008). *Image description using the complexity of its local structures.*
  ICPR. Uses recursive subdivision complexity as an image descriptor — the direct conceptual
  ancestor of the tree-grid feature here.

---

## Known limitations

**Resolution sensitivity.** The quadtree is built with a fixed `leaf_size` in pixels, so
complexity scores and tree depth vary with image resolution. Images of very different sizes
should be normalised before comparison, or the same resolution dataset should be used for
training and inference.

**JPEG compression artifacts.** Heavily compressed JPEGs inflate compression-ratio scores
in low-complexity regions (smooth areas accumulate block artifacts). This can make compressed
real photos look more like AI images under the compression method. Using `--method shannon`
or ensuring consistent JPEG quality across the dataset mitigates this.

**Cropped or composited subjects.** The alpha-channel background detection works well for
images with clean cutouts, but degrades for images where the subject bleeds into a complex
background. In those cases, background leaf complexity contaminates the subject feature vector.

**Generative model drift.** The spatial complexity signatures of AI generators change with
architecture. A classifier trained on outputs from one model family (e.g. Stable Diffusion 1.x)
may not generalise to newer architectures (FLUX, SD3, Midjourney v6) without retraining on
updated examples. The feature set is architecture-agnostic, but the decision boundary is not.

**Class imbalance and dataset bias.** Permutation importance and classifier accuracy are both
sensitive to the composition of the training set. A dataset dominated by one image subject
type (e.g. portraits) will produce a classifier that may not generalise to landscapes or
product shots. The `--balance` flag helps with class size imbalance but not subject-type bias.

**Photoshopping is a spectrum.** Minor retouching (skin smoothing, colour grading) produces
subtle signals that overlap with natural variation in real photos. Heavy compositing is easier
to detect. The three-class problem (authentic / synthetic / manipulated) is significantly harder than
binary real vs AI, and current accuracy on the photoshopped class tends to be lower.

**`max_complexity` is a dead feature.** Because the quadtree always contains at least one
maximally complex leaf, `max_complexity` is almost always 1.0 for every image regardless of
class. It should be excluded from classifier training (or will be automatically dropped by
`--prune-features 0.0`).

---

## Supported image formats

`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

RGBA images are supported — transparent regions are detected via the alpha channel and
excluded from complexity scoring and feature statistics.