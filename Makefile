# Makefile — Entropy Quadtree Pipeline
#
# Usage:
#   make phase1                 # collect samples and tune thresholds
#   make phase2                 # full feature extraction
#   make phase3                 # train anchor models + cross-tests
#   make phase4                 # combined best model
#   make all                    # run all phases in sequence
#   make clean-features         # delete extracted CSVs (re-run phase2)
#   make clean-models           # delete saved model bundles
#   make clean-results          # delete all results except examples/
#   make info                   # print directory sizes
#
# Prerequisites:
#   pip install pillow numpy scikit-learn matplotlib joblib kaggle requests
#
# Threshold values — read automatically from results/tuning/best_thresholds.json
# after running 'make phase1'. Falls back to 0 (no pruning) if not yet tuned.
# The JSON is written by tune_thresholds.py and keyed as {name}_{method}.
THRESHOLDS_JSON := results/tuning/best_thresholds.json

_threshold = $(shell python3 -c \
    "import json; d=json.load(open('$(THRESHOLDS_JSON)')); print(int(d.get('$1', 0)))" \
    2>/dev/null || echo 0)

SHANNON_THRESHOLD                 := $(call _threshold,stylegan_v1_shannon)
COMPRESSION_THRESHOLD             := $(call _threshold,stylegan_v1_compression)
DEEPFACELAB_SHANNON_THRESHOLD     := $(call _threshold,deepfacelab_shannon)
DEEPFACELAB_COMPRESSION_THRESHOLD := $(call _threshold,deepfacelab_compression)

# Common flags
RESIZE   := 256
PYTHON   := python3
SRC      := src

# Feature output directory
FEAT     := results/features
MODELS   := results/models

# ── Phony targets ─────────────────────────────────────────────────────────────

.PHONY: all phase1 phase2 phase3 phase4 \
        samples tune-stylegan tune-deepfacelab \
        clean-features clean-models clean-results \
        show-thresholds info help

all: phase1 phase2 phase3 phase4

# ── Phase 1: Sample collection and threshold tuning ──────────────────────────

phase1: samples tune-stylegan tune-deepfacelab

samples: \
    data/sample/stylegan_v1_synthetic \
    data/sample/FFHQ_authentic \
    data/sample/deepfacelab_manipulated \
    data/sample/stylegan_v1v2_synthetic \
    data/sample/sd14_synthetic \
    data/sample/photoshop_liquify_manipulated

data/sample/stylegan_v1_synthetic:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset xhlulu/140k-real-and-fake-faces \
	    --prefix real_vs_fake/real-vs-fake/train \
	    --classes fake --labels synthetic \
	    --label-detail stylegan_v1_portrait \
	    --methods shannon compression \
	    --resize $(RESIZE) --name stylegan_v1 \
	    --max-images 300 --save-sample 300

data/sample/FFHQ_authentic:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset xhlulu/140k-real-and-fake-faces \
	    --prefix real_vs_fake/real-vs-fake/train \
	    --classes real --labels authentic \
	    --label-detail real_portrait \
	    --methods shannon compression \
	    --resize $(RESIZE) --name FFHQ \
	    --max-images 300 --save-sample 300

data/sample/deepfacelab_manipulated:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset manjilkarki/deepfake-and-real-images \
	    --prefix Dataset/Train \
	    --classes Fake --labels manipulated \
	    --label-detail deepfacelab_swap \
	    --methods shannon compression \
	    --resize $(RESIZE) --name deepfacelab \
	    --max-images 300 --save-sample 300

data/sample/stylegan_v1v2_synthetic:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset kshitizbhargava/deepfake-face-images \
	    --prefix "Final Dataset" \
	    --classes Fake --labels synthetic \
	    --label-detail stylegan_v1v2_portrait \
	    --methods shannon compression \
	    --resize $(RESIZE) --name stylegan_v1v2 \
	    --max-images 300 --save-sample 300

data/sample/sd14_synthetic:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset bwandowando/faces-dataset-using-stable-diffusion-v14 \
	    --prefix Training \
	    --classes Female Male --labels synthetic synthetic \
	    --label-detail sd14_portrait sd14_portrait \
	    --methods shannon compression \
	    --resize $(RESIZE) --name sd14 \
	    --max-images 300 --save-sample 300

data/sample/photoshop_liquify_manipulated:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset tbourton/photoshopped-faces \
	    --prefix "" \
	    --classes modified --labels manipulated \
	    --label-detail photoshop_liquify \
	    --methods shannon compression \
	    --resize $(RESIZE) --name photoshop_liquify \
	    --max-images 300 --save-sample 300

tune-stylegan: data/sample/stylegan_v1_synthetic data/sample/FFHQ_authentic
	$(PYTHON) $(SRC)/tune_thresholds.py \
	    --input data/sample/stylegan_v1_synthetic data/sample/FFHQ_authentic \
	    --labels synthetic authentic \
	    --method shannon --leaf-size 4 --max-images 300
	$(PYTHON) $(SRC)/tune_thresholds.py \
	    --input data/sample/stylegan_v1_synthetic data/sample/FFHQ_authentic \
	    --labels synthetic authentic \
	    --method compression --leaf-size 16 --max-images 300
	$(PYTHON) $(SRC)/tune_plots.py --folder results/tuning/stylegan_v1/

tune-deepfacelab: data/sample/deepfacelab_manipulated data/sample/FFHQ_authentic
	$(PYTHON) $(SRC)/tune_thresholds.py \
	    --input data/sample/deepfacelab_manipulated data/sample/FFHQ_authentic \
	    --labels manipulated authentic \
	    --method shannon --leaf-size 4 --max-images 300
	$(PYTHON) $(SRC)/tune_thresholds.py \
	    --input data/sample/deepfacelab_manipulated data/sample/FFHQ_authentic \
	    --labels manipulated authentic \
	    --method compression --leaf-size 16 --max-images 300
	$(PYTHON) $(SRC)/tune_plots.py --folder results/tuning/deepfacelab/

# ── Phase 2: Full feature extraction ──────────────────────────────────────────
# Max-images is set well above each dataset's actual size so all images are
# processed. Chunk-size controls peak memory — lower it if you hit OOM errors.
# Dataset sizes (approximate):
#   xhlulu real train:    50,000    xhlulu fake train:   70,000
#   kshitizbhargava Fake: 10,000    bwandowando:        ~30,000 (Female+Male)
#   manjilkarki Fake:      4,000    tbourton modified:   ~1,000

CHUNK := 1000

phase2: \
    $(FEAT)/FFHQ_shannon.csv \
    $(FEAT)/stylegan_v1_shannon.csv \
    $(FEAT)/deepfacelab_shannon.csv \
    $(FEAT)/stylegan_v1v2_shannon.csv \
    $(FEAT)/sd14_shannon.csv \
    $(FEAT)/photoshop_liquify_shannon.csv

$(FEAT)/FFHQ_shannon.csv:
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset xhlulu/140k-real-and-fake-faces \
	    --prefix real_vs_fake/real-vs-fake/train \
	    --classes real --labels authentic \
	    --label-detail real_portrait \
	    --methods shannon compression \
	    --thresholds $(SHANNON_THRESHOLD) $(COMPRESSION_THRESHOLD) \
	    --resize $(RESIZE) --name FFHQ \
	    --max-images 999999 --chunk-size $(CHUNK)

$(FEAT)/stylegan_v1_shannon.csv: $(FEAT)/FFHQ_shannon.csv
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset xhlulu/140k-real-and-fake-faces \
	    --prefix real_vs_fake/real-vs-fake/train \
	    --classes fake --labels synthetic \
	    --label-detail stylegan_v1_portrait \
	    --methods shannon compression \
	    --thresholds $(SHANNON_THRESHOLD) $(COMPRESSION_THRESHOLD) \
	    --resize $(RESIZE) --name stylegan_v1 \
	    --max-images 999999 --chunk-size $(CHUNK) \
	    --pair-with $(FEAT)/FFHQ_shannon.csv $(FEAT)/FFHQ_compression.csv

$(FEAT)/deepfacelab_shannon.csv: $(FEAT)/FFHQ_shannon.csv
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset manjilkarki/deepfake-and-real-images \
	    --prefix Dataset/Train \
	    --classes Fake --labels manipulated \
	    --label-detail deepfacelab_swap \
	    --methods shannon compression \
	    --thresholds $(DEEPFACELAB_SHANNON_THRESHOLD) $(DEEPFACELAB_COMPRESSION_THRESHOLD) \
	    --resize $(RESIZE) --name deepfacelab \
	    --max-images 999999 --chunk-size $(CHUNK) \
	    --pair-with $(FEAT)/FFHQ_shannon.csv $(FEAT)/FFHQ_compression.csv

$(FEAT)/stylegan_v1v2_shannon.csv: $(FEAT)/FFHQ_shannon.csv
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset kshitizbhargava/deepfake-face-images \
	    --prefix "Final Dataset" \
	    --classes Fake --labels synthetic \
	    --label-detail stylegan_v1v2_portrait \
	    --methods shannon compression \
	    --thresholds $(SHANNON_THRESHOLD) $(COMPRESSION_THRESHOLD) \
	    --resize $(RESIZE) --name stylegan_v1v2 \
	    --max-images 999999 --chunk-size $(CHUNK) \
	    --pair-with $(FEAT)/FFHQ_shannon.csv $(FEAT)/FFHQ_compression.csv

$(FEAT)/sd14_shannon.csv: $(FEAT)/FFHQ_shannon.csv
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset bwandowando/faces-dataset-using-stable-diffusion-v14 \
	    --prefix Training \
	    --classes Female Male --labels synthetic synthetic \
	    --label-detail sd14_portrait sd14_portrait \
	    --methods shannon compression \
	    --thresholds $(SHANNON_THRESHOLD) $(COMPRESSION_THRESHOLD) \
	    --resize $(RESIZE) --name sd14 \
	    --max-images 999999 --chunk-size $(CHUNK) \
	    --pair-with $(FEAT)/FFHQ_shannon.csv $(FEAT)/FFHQ_compression.csv

$(FEAT)/photoshop_liquify_shannon.csv: $(FEAT)/FFHQ_shannon.csv
	$(PYTHON) $(SRC)/stream_batch.py \
	    --dataset tbourton/photoshopped-faces \
	    --prefix "" \
	    --classes modified --labels manipulated \
	    --label-detail photoshop_liquify \
	    --methods shannon compression \
	    --thresholds $(DEEPFACELAB_SHANNON_THRESHOLD) $(DEEPFACELAB_COMPRESSION_THRESHOLD) \
	    --resize $(RESIZE) --name photoshop_liquify \
	    --max-images 999999 --chunk-size $(CHUNK) \
	    --pair-with $(FEAT)/FFHQ_shannon.csv $(FEAT)/FFHQ_compression.csv

# ── Phase 3: Train anchor models and cross-test ───────────────────────────────

phase3: \
    $(MODELS)/stylegan_v1_shannon.joblib \
    $(MODELS)/stylegan_v1_compression.joblib \
    $(MODELS)/deepfacelab_shannon.joblib \
    $(MODELS)/deepfacelab_compression.joblib \
    cross-tests

$(MODELS)/stylegan_v1_shannon.joblib: $(FEAT)/stylegan_v1_shannon.csv
	$(PYTHON) $(SRC)/classify.py $(FEAT)/stylegan_v1_shannon.csv \
	    --fast --balance --prune-features 0.0 \
	    --save-model stylegan_v1_shannon \
	    --method shannon --leaf-size 4 --resize $(RESIZE)

$(MODELS)/stylegan_v1_compression.joblib: $(FEAT)/stylegan_v1_compression.csv
	$(PYTHON) $(SRC)/classify.py $(FEAT)/stylegan_v1_compression.csv \
	    --fast --balance --prune-features 0.0 \
	    --save-model stylegan_v1_compression \
	    --method compression --leaf-size 16 --resize $(RESIZE)

$(MODELS)/deepfacelab_shannon.joblib: $(FEAT)/deepfacelab_shannon.csv
	$(PYTHON) $(SRC)/classify.py $(FEAT)/deepfacelab_shannon.csv \
	    --fast --balance --prune-features 0.0 \
	    --save-model deepfacelab_shannon \
	    --method shannon --leaf-size 4 --resize $(RESIZE)

$(MODELS)/deepfacelab_compression.joblib: $(FEAT)/deepfacelab_compression.csv
	$(PYTHON) $(SRC)/classify.py $(FEAT)/deepfacelab_compression.csv \
	    --fast --balance --prune-features 0.0 \
	    --save-model deepfacelab_compression \
	    --method compression --leaf-size 16 --resize $(RESIZE)

cross-tests: \
    $(FEAT)/stylegan_v1_shannon.csv \
    $(FEAT)/stylegan_v1v2_shannon.csv \
    $(FEAT)/sd14_shannon.csv \
    $(FEAT)/deepfacelab_shannon.csv \
    $(FEAT)/photoshop_liquify_shannon.csv
	# StyleGAN v1 → v1v2 and SD14
	$(PYTHON) $(SRC)/classify.py $(FEAT)/stylegan_v1_shannon.csv \
	    --test-csv $(FEAT)/stylegan_v1v2_shannon.csv --fast --balance
	$(PYTHON) $(SRC)/classify.py $(FEAT)/stylegan_v1_shannon.csv \
	    --test-csv $(FEAT)/sd14_shannon.csv --fast --balance
	$(PYTHON) $(SRC)/classify.py $(FEAT)/stylegan_v1_compression.csv \
	    --test-csv $(FEAT)/stylegan_v1v2_compression.csv --fast --balance
	$(PYTHON) $(SRC)/classify.py $(FEAT)/stylegan_v1_compression.csv \
	    --test-csv $(FEAT)/sd14_compression.csv --fast --balance
	# DeepFaceLab → Photoshop Liquify
	$(PYTHON) $(SRC)/classify.py $(FEAT)/deepfacelab_shannon.csv \
	    --test-csv $(FEAT)/photoshop_liquify_shannon.csv --fast --balance
	$(PYTHON) $(SRC)/classify.py $(FEAT)/deepfacelab_compression.csv \
	    --test-csv $(FEAT)/photoshop_liquify_compression.csv --fast --balance

# ── Phase 4: Combined best model ──────────────────────────────────────────────

phase4: $(MODELS)/combined_shannon.joblib $(MODELS)/combined_compression.joblib

$(MODELS)/combined_shannon.joblib: phase2
	$(PYTHON) $(SRC)/classify.py \
	    $(FEAT)/stylegan_v1_shannon.csv \
	    $(FEAT)/stylegan_v1v2_shannon.csv \
	    $(FEAT)/sd14_shannon.csv \
	    $(FEAT)/deepfacelab_shannon.csv \
	    $(FEAT)/photoshop_liquify_shannon.csv \
	    --fast --balance --prune-features 0.0 \
	    --save-model combined_shannon \
	    --method shannon --leaf-size 4 --resize $(RESIZE)

$(MODELS)/combined_compression.joblib: phase2
	$(PYTHON) $(SRC)/classify.py \
	    $(FEAT)/stylegan_v1_compression.csv \
	    $(FEAT)/stylegan_v1v2_compression.csv \
	    $(FEAT)/sd14_compression.csv \
	    $(FEAT)/deepfacelab_compression.csv \
	    $(FEAT)/photoshop_liquify_compression.csv \
	    --fast --balance --prune-features 0.0 \
	    --save-model combined_compression \
	    --method compression --leaf-size 16 --resize $(RESIZE)

# ── Maintenance ───────────────────────────────────────────────────────────────

show-thresholds:
	@echo "── Thresholds (from $(THRESHOLDS_JSON)) ──"
	@echo "  stylegan_v1   shannon:     $(SHANNON_THRESHOLD)"
	@echo "  stylegan_v1   compression: $(COMPRESSION_THRESHOLD)"
	@echo "  deepfacelab   shannon:     $(DEEPFACELAB_SHANNON_THRESHOLD)"
	@echo "  deepfacelab   compression: $(DEEPFACELAB_COMPRESSION_THRESHOLD)"
	@if [ ! -f $(THRESHOLDS_JSON) ]; then \
	    echo "  (file not found — run 'make phase1' to generate)"; fi

clean-features:
	rm -f results/features/*.csv results/features/*.json
	@echo "Feature CSVs deleted. Re-run: make phase2"

clean-models:
	rm -f results/models/*.joblib
	@echo "Model bundles deleted. Re-run: make phase3 phase4"

clean-results:
	rm -rf results/features results/models results/classify \
	       results/scatter results/spatial results/depth \
	       results/overlays results/predictions results/tuning
	@echo "All results deleted (results/examples/ preserved)."

info:
	@echo "── Directory sizes ──"
	@du -sh data/sample/*       2>/dev/null | sort -h || echo "  data/sample/: empty"
	@du -sh results/features/*  2>/dev/null | sort -h || echo "  results/features/: empty"
	@du -sh results/models/*    2>/dev/null | sort -h || echo "  results/models/: empty"
	@du -sh results/            2>/dev/null | tail -1

help:
	@echo "Entropy Quadtree Pipeline"
	@echo ""
	@echo "  make phase1          collect samples + tune thresholds"
	@echo "  make phase2          full feature extraction (requires phase1)"
	@echo "  make phase3          train models + cross-tests (requires phase2)"
	@echo "  make phase4          combined model (requires phase2)"
	@echo "  make all             run all phases"
	@echo "  make show-thresholds show thresholds read from best_thresholds.json"
	@echo ""
	@echo "  make clean-features  delete extracted CSVs"
	@echo "  make clean-models    delete model bundles"
	@echo "  make clean-results   delete all results (keeps examples/)"
	@echo "  make info            print directory sizes"
	@echo ""
	@echo "  Thresholds are read automatically from $(THRESHOLDS_JSON)"
	@echo "  after running 'make phase1'. Run 'make show-thresholds' to verify."