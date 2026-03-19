# Alytes-ReID

Individual re-identification of **Alytes obstetricans** (common midwife toad) from dorsal photographs.

The system supports mark-recapture population studies by automatically matching toad photographs to known individuals based on unique dorsal skin patterns (wart arrangements, pigmentation patterns).

---

## Quick Start

### Google Colab Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| **Setup & Training** | Download data, train detection & re-ID models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/Alytes-ReID/blob/claude/alytes-reid-system-qgved/notebooks/01_setup_and_training.ipynb) |
| **Toad Re-ID Tool** | Upload a photo → get individual match | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/Alytes-ReID/blob/claude/alytes-reid-system-qgved/notebooks/02_toad_reid.ipynb) |

### Gradio Web App

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL shown in the terminal. The app lets you:
- Upload a toad photo and get the top matching individuals
- Browse the database of known individuals
- Register new individuals
- Export results as CSV for mark-recapture analysis (RMark/MARK compatible)

---

## How It Works

```
Input Image → [Detection] → [Segmentation] → [Preprocessing] → [Re-ID] → Match
                YOLOv8         SAM2            Align + Norm      Metric Learning
```

### Module 1: Detection & Segmentation
- **YOLO11** detects the toad bounding box in the photo
- **SAM2** (Segment Anything Model 2) generates a precise segmentation mask using the bounding box as prompt
- Why two-stage: SAM2 works well zero-shot, produces superior masks, and requires less annotation

### Module 2: Preprocessing
- Tight crop around the segmented toad
- Background removal (mask applied)
- Pose alignment via PCA (head-tail axis → vertical)
- Color normalization with CLAHE
- Resize to 256x256 maintaining aspect ratio

### Module 3: Re-Identification
- EfficientNet-B0 backbone with ArcFace metric learning
- Extracts 512-dim embedding per individual
- FAISS-powered similarity search against the database
- Confidence threshold flags potential new individuals

---

## Project Structure

```
Alytes-ReID/
├── app.py                     # Gradio web interface
├── config/
│   ├── detection.yaml         # YOLOv8 training config
│   ├── preprocessing.yaml     # Preprocessing parameters
│   └── reid.yaml              # Re-ID model config
├── src/
│   ├── data/                  # Data download & preparation
│   ├── detection/             # YOLOv8 training, prediction, evaluation
│   ├── segmentation/          # SAM2 segmentation wrapper
│   ├── preprocessing/         # Alignment, normalization, pipeline
│   ├── reid/                  # Model, training, database, matching
│   └── utils/                 # Visualization, I/O helpers
├── notebooks/
│   ├── 01_setup_and_training.ipynb
│   └── 02_toad_reid.ipynb
├── tests/                     # Unit tests
├── data/                      # Local data (gitignored)
└── docs/
```

---

## Installation

```bash
git clone https://github.com/danort92/Alytes-ReID.git
cd Alytes-ReID
pip install -r requirements.txt
```

For development:
```bash
pip install -e ".[dev,reid,ui]"
```

### Requirements
- Python 3.10+
- PyTorch 2.0+
- CUDA-compatible GPU recommended (works on CPU too)

---

## Data

### Public Data (for detection training)
- **Roboflow frogs dataset**: ~350 annotated images including Alytes
- **iNaturalist**: Alytes obstetricans images via API (research-grade)

```bash
# Download iNaturalist images
python -m src.data.download_inat --output data/raw/inaturalist --max-images 1000
```

### Private Data (for re-ID)
Labeled photos with individual IDs provided by the field biologist.

---

## Usage

### 1. Train Detection Model
```bash
python -m src.detection.train --config config/detection.yaml
```

### 2. Run Detection + Segmentation
```bash
python -m src.detection.predict --model data/models/detection/best.pt --input photo.jpg
python -m src.segmentation.segment --image photo.jpg --bbox 100 50 400 350
```

### 3. Preprocess
```bash
python -m src.preprocessing.pipeline --config config/preprocessing.yaml --input cropped.png --mask mask.png
```

### 4. Match Against Database
```bash
python -m src.reid.match --config config/reid.yaml --model data/models/reid/reid_model.pt --db data/models/reid --image preprocessed.png
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Detection | Ultralytics YOLO11 |
| Segmentation | SAM2 |
| Re-ID backbone | timm (EfficientNet-B0) |
| Metric learning | pytorch-metric-learning (ArcFace) |
| Similarity search | FAISS |
| Web interface | Gradio |
| Experiment tracking | MLflow |
| Image processing | OpenCV, albumentations |

---

## References

- [SAM2](https://github.com/facebookresearch/segment-anything-2)
- [HotSpotter](https://github.com/Erotemic/hotspotter) — wildlife re-ID baseline
- [WildMe / Wildbook](https://www.wildme.org/)
- [pytorch-metric-learning](https://github.com/KevinMusgrave/pytorch-metric-learning)
- [iNaturalist API](https://api.inaturalist.org/v1/docs/)

---

## License

This project is for academic and conservation research purposes.
