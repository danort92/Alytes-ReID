# Alytes-ReID: Individual Re-Identification of Midwife Toads

## Project Overview

A computer vision pipeline for individual re-identification of *Alytes obstetricans* (common midwife toad) from dorsal photographs. The system supports mark-recapture population studies by automatically matching toad photographs to known individuals based on unique dorsal skin patterns (wart arrangements, pigmentation patterns).

**End goal**: A user-friendly tool where a researcher uploads a photo of a toad and gets back: (1) the segmented toad image, (2) the best matching individual(s) from the database with confidence scores, and (3) the option to register a new individual.

## Architecture

Three independent modules, developed sequentially:

```
Input Image → [Module 1: Detection/Segmentation] → Cropped toad
            → [Module 2: Preprocessing] → Standardized dorsal image
            → [Module 3: Re-ID] → Individual match + confidence
```

## Module 1: Toad Detection & Segmentation (START HERE)

### Goal
Given an image (variable background, lighting, quality), detect the toad and produce a clean binary mask + cropped image.

### Approach
Two-stage pipeline:
1. **Detection**: YOLOv8n or YOLOv8s for bounding box detection (lightweight, fast)
2. **Segmentation**: SAM2 (Segment Anything Model 2) prompted with the YOLO bounding box for precise mask generation

Why two-stage instead of YOLOv8-seg:
- SAM2 works well zero-shot — less annotation needed
- More flexible with variable backgrounds
- Easier to swap detection model later
- Mask quality from SAM2 is superior for downstream re-ID

### Data Strategy (Module 1)
Use public data to build and test detection before Schmidt's data arrives:
- **Roboflow "frogs" dataset** (~350 annotated images including Alytes): https://universe.roboflow.com/brad-dwyer/frogs-agkpn
- **iNaturalist**: Download Alytes images via API or inat_downloader script
  - Filter: genus Alytes, quality_grade=research, with photos
  - Target: 500-1000 images for detection training
- **Manual annotation**: Use Roboflow or Label Studio for any additional bbox annotations

### Deliverables Module 1
- [ ] Data download and preparation script (`src/data/download_inat.py`)
- [ ] YOLOv8 fine-tuning script (`src/detection/train.py`)
- [ ] SAM2 segmentation wrapper (`src/segmentation/segment.py`)
- [ ] Evaluation script with IoU metrics (`src/detection/evaluate.py`)
- [ ] Demo notebook showing detection → segmentation pipeline

## Module 2: Preprocessing & Standardization

### Goal
Normalize segmented toad images for consistent re-ID comparison.

### Steps
1. **Tight crop**: Crop to mask bounding box with small padding
2. **Background removal**: Apply mask, set background to black or white
3. **Pose alignment**: Rotate so head-tail axis is vertical
   - Option A: PCA on mask pixels (fast, simple)
   - Option B: Keypoint detection for head/tail (more robust)
4. **Resize**: Fixed resolution (e.g., 224x224 or 256x256)
5. **Color normalization**: Histogram equalization or CLAHE to handle lighting variation
6. **Optional**: Extract dorsal region only (exclude limbs) for cleaner pattern matching

### Deliverables Module 2
- [ ] Preprocessing pipeline (`src/preprocessing/pipeline.py`)
- [ ] Visual QA notebook showing before/after standardization
- [ ] Config file for preprocessing parameters

## Module 3: Individual Re-Identification

### Goal
Given a standardized dorsal image, find the matching individual from a database or flag as new.

### Approach: Metric Learning
- **Backbone**: ResNet50 or EfficientNet-B0, pretrained on ImageNet
- **Head**: ArcFace loss (better than triplet loss for open-set recognition)
- **Training**: Fine-tune on Schmidt's labeled data (photos with individual IDs)
- **Inference**: Extract embedding → cosine similarity search against database
- **Threshold**: Below threshold → flag as potential new individual

### Alternative/Complementary Approaches
- **HotSpotter**: Pattern-based matching using SIFT-like features on dorsal patterns. Worth benchmarking as a baseline — no training needed.
- **Siamese Network**: If dataset is very small (<100 individuals)
- **LOFTR / SuperGlue**: Local feature matching for verification

### Deliverables Module 3
- [ ] Training script with metric learning (`src/reid/train.py`)
- [ ] Embedding extraction and database management (`src/reid/database.py`)
- [ ] Matching/query script (`src/reid/match.py`)
- [ ] Evaluation with CMC curves and mAP

## User Interface

### Framework
Gradio (simpler for ML demos, easy sharing via public link, built-in image upload)

### Features
- Upload single image → get segmented toad + top-K matches with confidence
- Upload batch → process multiple images
- Database management: view all individuals, add new ones
- Export results as CSV for mark-recapture analysis (compatible with RMark/MARK)
- Simple enough for a field biologist with no coding experience

### Deliverables UI
- [ ] Gradio app (`app.py`)
- [ ] Database viewer page
- [ ] Batch processing page
- [ ] Export functionality

## Project Structure

```
Alytes-ReID/
├── CLAUDE.md                  # This file
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── app.py                     # Gradio web interface
├── config/
│   ├── detection.yaml         # YOLO training config
│   ├── preprocessing.yaml     # Preprocessing parameters
│   └── reid.yaml              # Re-ID model config
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── download_inat.py   # iNaturalist data download
│   │   ├── download_roboflow.py
│   │   └── prepare_dataset.py # Format conversion, splits
│   ├── detection/
│   │   ├── train.py           # YOLOv8 fine-tuning
│   │   ├── predict.py         # Run detection on new images
│   │   └── evaluate.py        # Detection metrics
│   ├── segmentation/
│   │   ├── segment.py         # SAM2 segmentation
│   │   └── utils.py           # Mask utilities
│   ├── preprocessing/
│   │   ├── pipeline.py        # Full preprocessing pipeline
│   │   ├── align.py           # Pose alignment
│   │   └── normalize.py       # Color/histogram normalization
│   ├── reid/
│   │   ├── model.py           # Re-ID model architecture
│   │   ├── train.py           # Training with metric learning
│   │   ├── database.py        # Embedding database management
│   │   └── match.py           # Query matching
│   └── utils/
│       ├── visualization.py   # Plotting utilities
│       └── io.py              # File I/O helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_detection_demo.ipynb
│   ├── 03_segmentation_demo.ipynb
│   ├── 04_preprocessing_demo.ipynb
│   └── 05_reid_demo.ipynb
├── tests/
│   ├── test_detection.py
│   ├── test_segmentation.py
│   ├── test_preprocessing.py
│   └── test_reid.py
├── data/                      # .gitignore this
│   ├── raw/
│   ├── processed/
│   └── models/
└── docs/
    └── pipeline_diagram.md
```

## Tech Stack

- **Python 3.10+**
- **PyTorch 2.0+** — deep learning framework
- **Ultralytics** — YOLOv8 detection
- **segment-anything-2** — SAM2 segmentation
- **pytorch-metric-learning** — ArcFace, triplet loss, miners
- **timm** — pretrained backbones (EfficientNet, ResNet)
- **Gradio** — web interface
- **MLflow** — experiment tracking
- **OpenCV** — image processing
- **albumentations** — data augmentation
- **FAISS** — fast similarity search for embeddings
- **pytest** — testing

## Development Guidelines

- Type hints on all functions
- Docstrings (Google style)
- Each module should work independently and be testable in isolation
- Config-driven: no hardcoded paths or hyperparameters
- All experiments tracked with MLflow
- GPU training code should also work on CPU (slower) for testing
- Use pathlib for all file paths
- Logging with Python's logging module, not print statements

## Current Status

**Phase**: Module 1 — Detection & Segmentation
**Next step**: Set up repo structure, download public Alytes images from iNaturalist, train YOLOv8 detector.

## Key References

- SAM2: https://github.com/facebookresearch/segment-anything-2
- HotSpotter (wildlife re-ID): https://github.com/Erotemic/hotspotter
- WildMe / Wildbook: https://www.wildme.org/
- pytorch-metric-learning: https://github.com/KevinMusgrave/pytorch-metric-learning
- iNaturalist API: https://api.inaturalist.org/v1/docs/
- Roboflow frogs dataset: https://universe.roboflow.com/brad-dwyer/frogs-agkpn
