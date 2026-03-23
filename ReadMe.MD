# Quadtree Complexity Analysis for Image Forensics

Detect AI-generated, face-swapped, and manipulated images by analysing how pixel-level complexity distributes spatially across a quadtree decomposition of the image. Authentic photographs show high, heterogeneous complexity driven by natural scene statistics; synthetic images tend to be smoother and more locally uniform, even when their global statistics look similar.

---

## How it works

A quadtree is built by recursively splitting an image into quadrants and scoring each region for complexity (Shannon entropy, zlib compression ratio, or pixel variance). Splitting stops when a region reaches the target leaf size. The resulting tree is summarised into a fixed-length feature vector — global statistics, boundary-jump signals, merge-step deltas, per-channel scores, and a 16 × 16 spatial grid — and fed to a classifier (Logistic Regression, SVM, Random Forest, or Gradient Boosting).

The spatial grid is extracted at tree depth 4 (4⁴ = 256 cells) using Morton order, making it resolution-independent and directly comparable across datasets when a consistent `--resize` is used.

---

## Pipeline overview

```
Images  →  batch.py / stream_batch.py     →  features CSV
                                          →  merge_features.py  (optional multi-method merge)
                                          →  classify.py         →  report + model bundle
                                          →  predict.py          →  per-image predictions
```

Supporting tools: `tune_thresholds.py`, `tune_plots.py`, `scatter.py`, `depth_distribution.py`, `grid_importance.py`, `main.py` (single-image overlay).

---

## Installation

**Requirements:** Python ≥ 3.10

```bash
pip install -r requirements.txt
```

Or install as a package (enables the `qia-*` entry points):

```bash
pip install -e .
```

### Dependencies

| Package | Version |
|---|---|
| Pillow | ≥ 10.0 |
| numpy | ≥ 1.24 |
| scikit-learn | ≥ 1.3 |
| joblib | ≥ 1.3 |
| requests | ≥ 2.31 |
| matplotlib | ≥ 3.7 |

---

## Quick start

### 1. Single-image overlay

Visualise the complexity map for one image:

```bash
python3 src/main.py photo.jpg
python3 src/main.py photo.jpg --method compression --legend --borders
```

Output: `photo_complexity.png` — the original image with a colour overlay (blue = low complexity, red = high).

### 2. Extract features from a dataset

```bash
python3 src/batch.py \
    --input real_photos/ ai_images/ \
    --labels authentic synthetic \
    --output results/features/ \
    --method shannon --resize 256 \
    --name my_dataset
# → results/features/my_dataset_shannon.csv
```

Always use `--resize` when combining images from different sources so that leaf sizes and spatial grid cells are comparable.

### 3. Train a classifier

```bash
python3 src/classify.py results/features/my_dataset_shannon.csv --fast --balance
```

Save the best model for later inference:

```bash
python3 src/classify.py results/features/my_dataset_shannon.csv \
    --fast --balance \
    --method shannon --leaf-size 4 \
    --save-model my_dataset_shannon
# → results/models/my_dataset_shannon.joblib
```

### 4. Classify new images

```bash
python3 src/predict.py results/models/my_dataset_shannon.joblib new_images/
python3 src/predict.py results/models/my_dataset_shannon.joblib photo.jpg \
    --output results/predictions/photo.csv --overlay
```

---

## Scripts

### `main.py` — single-image complexity overlay

```
python3 src/main.py <image> [options]

--method    shannon | compression | variance  (default: shannon)
--leaf_size int                               (default: 4)
--threshold float                             percentile pruning, default off
--alpha     int                               overlay opacity 0-255 (default: 120)
--borders                                     show quadrant outlines
--legend                                      append colorbar below image
```

### `batch.py` — batch feature extraction

```
python3 src/batch.py --input <folder(s)> --labels <label(s)> --output <path> [options]

--method      shannon | compression | variance  (default: shannon)
--leaf-size   int                               (default: 4)
--resize      int   resize every image to NxN  (recommended: 256)
--threshold   float percentile pruning cutoff
--name        str   dataset name prefix for output filename
--max-images  int   cap images per folder
--append            append to existing CSV
```

Multiple `--input` folders and matching `--labels` can be given in one run to build a labeled dataset in a single pass.

### `stream_batch.py` — stream directly from Kaggle

Downloads a Kaggle dataset zip into memory, processes images chunk-by-chunk (never writes images to disk), and outputs feature CSVs. Supports multiple methods in a single download pass.

```
python3 src/stream_batch.py \
    --dataset ciplab/real-and-fake-face-detection \
    --classes training_fake training_real \
    --labels manipulated authentic \
    --methods shannon compression \
    --resize 256 --name ciplab_faces
```

### `classify.py` — train and evaluate classifiers

```
python3 src/classify.py <features.csv> [options]

--fast                  Logistic Regression + Random Forest only (seconds)
--balance               undersample majority class
--test-size float       train/test split fraction (default: 0.2)
--cv int                cross-validation folds (default: 5)
--prune-features float  drop features below this permutation importance threshold
--permutation-importance  compute permutation importance for RF
--test-csv PATH         cross-dataset eval: train on CSV, test on this separate CSV
--save-model NAME       save model bundle to results/models/{NAME}.joblib
--method / --leaf-size  required when using --save-model
```

Classifiers: Logistic Regression, Linear SVM, Random Forest, Gradient Boosting. `--fast` uses only LR and RF.

### `predict.py` — inference with a saved model

```
python3 src/predict.py <model.joblib> <image(s) or folder> [options]

--output PATH       save predictions CSV
--overlay           save complexity overlay PNG per image
--overlay-dir DIR   folder for overlay PNGs (default: results/overlays/)
--quiet             suppress per-image console output
--info              print model bundle metadata and exit
```

### `merge_features.py` — combine features from multiple methods

Merges per-method CSVs into a single wide CSV with prefixed columns and cross-method ratio/delta features (e.g. `cross_shannon_compression_mean_complexity_ratio`). These cross-method features are often the strongest discriminators because they directly capture the divergence between structural repetition and tonal entropy.

```
python3 src/merge_features.py \
    --inputs results/features/shannon.csv results/features/compression.csv \
    --methods shannon compression \
    --output results/features/merged_shannon_compression.csv
```

### `tune_thresholds.py` — tune the pruning threshold

Builds each image's quadtree once, then re-extracts features at each candidate threshold in memory. Cross-validates a Random Forest at each value to find the optimal pruning percentile.

```
python3 src/tune_thresholds.py \
    --input real_photos/ ai_images/ \
    --labels authentic synthetic \
    --method compression --leaf_size 16
```

### `tune_plots.py` — visualise tuning results

Reads CSVs from `tune_thresholds.py` and produces three plots: accuracy vs threshold, leaf count vs threshold, and accuracy vs leaf count (coloured by threshold percentile).

```
python3 src/tune_plots.py
python3 src/tune_plots.py --folder results/tuning/stylegan_v1/
```

### `scatter.py` — feature scatter plots

Plots any two features coloured by class label. `--auto` mode ranks features by ANOVA F-statistic and greedily selects the top N that are mutually uncorrelated.

```
python3 src/scatter.py results/features/shannon.csv --auto --top-features 10
python3 src/scatter.py results/features/shannon.csv --x mean_complexity --y std_complexity
```

### `depth_distribution.py` — choose tree-grid depth

Samples images, builds trees, and reports the empirical depth distribution. Use this to confirm that your chosen `TREE_GRID_DEPTH` (default 4) is reliably reached so no spatial cells require parent-value propagation.

### `grid_importance.py` — spatial importance heatmap

Fits a Random Forest on a features CSV, computes permutation importance for every `tree_grid_*` cell, and renders a 16 × 16 heatmap showing which spatial regions drive classification. Optionally overlays the heatmap on a reference image.

```
python3 src/grid_importance.py results/features/compression.csv
python3 src/grid_importance.py results/features/compression.csv --image face.jpg
```

### `reproduce.py` — validate reproducibility

Checks the current environment against minimum version requirements and validates sidecar metadata from feature CSVs or model bundles. Can optionally reconstruct and re-run the exact `classify.py` command that produced a saved bundle.

```
python3 src/reproduce.py results/models/my_dataset_shannon.joblib
python3 src/reproduce.py results/models/my_dataset_shannon.joblib --rerun
python3 src/reproduce.py --all
```

---

## Features extracted

| Feature | Description |
|---|---|
| `mean_complexity` | Mean complexity across subject leaf nodes |
| `std_complexity` | Variance of complexity — low in AI images |
| `min_complexity` | Floor of the complexity distribution |
| `complexity_range` | max − min spread |
| `mean_boundary_delta` | Mean \|parent − child\| complexity at splits |
| `max_boundary_delta` | Peak boundary jump — manipulation signal |
| `mean_merge_delta` | Mean std(children) / (parent + ε) across splits |
| `max_merge_delta` | Peak merge delta — highlights manipulation boundaries |
| `std_merge_delta` | Variance of merge deltas — high in real, low in AI |
| `mean_complexity_r/g/b` | Per-channel mean complexity |
| `complexity_delta_rg/rb/gb` | Inter-channel complexity differences |
| `tree_grid_000–255` | 16 × 16 spatial complexity map in Morton order |

---

## Scoring methods

| Method | Description | Default leaf size |
|---|---|---|
| `shannon` | Normalised Shannon entropy of pixel values | 4 px |
| `compression` | zlib compression ratio — captures structural repetition | 16 px |
| `variance` | Normalised pixel variance — fastest, good compression proxy | 4 px |

Shannon and compression are partially independent: AI regions often have high Shannon entropy but low compression ratio because their "noise" is structurally repetitive. The `merge_features.py` cross-method ratios encode this divergence directly.

---

## Directory structure

```
results/
  features/     feature CSVs + sidecar JSON metadata
  models/       saved model bundles (.joblib)
  tuning/       threshold sweep CSVs and plots
  classify/     classification reports
  scatter/      scatter plot PNGs
  spatial/      spatial grid importance heatmaps
  depth/        depth distribution plots
  overlays/     complexity overlay PNGs
  predictions/  inference output CSVs
  examples/     example overlays (tracked in git)
```

All paths are configured centrally in `src/config.py`.

---

## Configuration (`src/config.py`)

Central defaults for the entire pipeline. Change a value here and it propagates to every script.

| Setting | Default | Description |
|---|---|---|
| `DEFAULT_RESIZE` | 256 | Resize applied to images before processing |
| `DEFAULT_METHOD` | `shannon` | Scoring method |
| `DEFAULT_LEAF_SIZES` | `{shannon:4, compression:16, variance:4}` | Per-method leaf sizes |
| `TREE_GRID_DEPTH` | 4 | Depth for spatial grid extraction (4⁴ = 256 cells) |
| `BG_THRESHOLD` | 0.95 | Transparent-pixel fraction above which a node is treated as background |

---

## Reproducibility

Each feature extraction run writes a sidecar JSON (e.g. `shannon.csv.json`) recording the method, leaf size, resize, threshold, dataset paths, and timestamp. Model bundles (`.joblib`) embed the same parameters alongside the fitted classifier and scaler, so `predict.py` can reconstruct the quadtree identically without any extra flags.

Use `reproduce.py` to validate that your environment meets the minimum dependency versions before re-running experiments.

---

## License

Apache 2.0