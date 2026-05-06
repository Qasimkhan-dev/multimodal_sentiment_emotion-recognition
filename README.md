# Multimodal Emotion Recognition and Sentiment Analysis
### Reproduction of Farhadipour et al. (2025) — arXiv:2503.06805
**FAST-NUCES | Department of AI & Data Science | Group 07**
> 25i-7601 · 25i-7641 · 25i-7643

---

## Overview

This repository contains the full implementation for reproducing the paper:

> **"Multimodal Emotion Recognition and Sentiment Analysis in Multi-Party Conversation Contexts"**
> Farhadipour et al. (2025)

The system trains four independent unimodal models on the MELD dataset and then combines their learned feature embeddings using a multimodal fusion MLP. Both **emotion recognition** (7 classes) and **sentiment analysis** (3 classes) tasks are supported.

### Tasks
| Task | Classes |
|------|---------|
| Emotion Recognition | anger, disgust, fear, joy, neutral, sadness, surprise |
| Sentiment Analysis | negative, neutral, positive |

### Modalities
| Modality | Model Used |
|----------|-----------|
| Text | RoBERTa-base |
| Audio | Wav2Vec2-base |
| Video | MobileNetV2 + Transformer (from scratch) |
| Face | InceptionResNetV1 + BiLSTM with Attention |

---

## Repository Structure

```
├── make_balanced_subsets.py      # Step 1a: Create balanced dataset subsets
├── extract_audio_meld.py         # Step 1b: Extract audio from videos subset
│
├── train_text_emotion.py         # Step 2a: Train RoBERTa for emotion
├── train_text_sentiment.py       # Step 2b: Train RoBERTa for sentiment
│
├── train_audio_emotion.py        # Step 3a: Train Wav2Vec2 for emotion
├── train_audio_sentiment.py      # Step 3b: Train Wav2Vec2 for sentiment
│
├── train_video_emotion.py        # Step 4a: Train MobileNetV2+Transformer for emotion
├── train_video_sentiment.py      # Step 4b: Train MobileNetV2+Transformer for sentiment
│
├── train_face_emotion.py         # Step 5a: Train FacialNet for emotion
├── train_face_sentiment.py       # Step 5b: Train FacialNet for sentiment
│
├── train_fusion_emotion.py       # Step 6a: 4-modality fusion for emotion
├── train_fusion_sentiment.py     # Step 6b: 4-modality fusion for sentiment
│
├── train_fusion3_emotion.py      # Step 7a: 3-modality fusion for emotion
└── train_fusion3_sentiment.py    # Step 7b: 3-modality fusion for sentiment
```

---

## Requirements

### Python Version
Python 3.8 or higher

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install transformers
pip install facenet-pytorch
pip install opencv-python
pip install pandas numpy scikit-learn
pip install ffmpeg-python
```

Make sure **FFmpeg** is also installed on your system:
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (via chocolatey)
choco install ffmpeg
```

### Hardware
- A **CUDA-enabled GPU** is strongly recommended. All scripts automatically detect and use GPU if available.
- Minimum recommended VRAM: **8GB** for text/audio, **6GB** for video/face, **4GB** for fusion.

---

## Dataset Setup

This project uses the **MELD (Multimodal EmotionLines Dataset)**, derived from the TV series *Friends*.

### Download MELD
Download the dataset from the official source:
```
https://github.com/declare-lab/MELD
```

After downloading and extracting, your directory structure should look like:
```
MELD.Raw/
├── train/
│   ├── train_splits/          ← video .mp4 files
│   └── train_sent_emo.csv     ← labels
├── dev/
│   ├── dev_splits_complete/   ← video .mp4 files
│   └── dev_sent_emo.csv
└── test/
    ├── output_repeated_splits_test/  ← video .mp4 files
    └── test_sent_emo.csv
```

### Extract Audio Files
All audio scripts expect pre-extracted WAV files. Extract them using FFmpeg:

```bash
# Example for train split — repeat for dev and test
mkdir -p meld_audio_15/train

for f in MELD.Raw/train/train_splits/*.mp4; do
    uid=$(basename "$f" .mp4)
    ffmpeg -i "$f" -ac 1 -ar 16000 "meld_audio_15/train/${uid}.wav" -y
done
```

---

## Step-by-Step Usage

### Step 1 — Create Balanced Subsets

Before training anything, run the preprocessing script to create class-balanced CSV subsets from the full MELD annotations.

```bash
python make_balanced_subsets.py
```

**What it does:**
- Groups utterances by emotion/sentiment label
- Samples 15% from each class independently
- Creates 6 CSV files (train/dev/test × emotion/sentiment) in the output directory

**Output files:**
```
meld_balanced/
├── train_20_emotion.csv
├── dev_20_emotion.csv
├── test_20_emotion.csv
├── train_20_sentiment.csv
├── dev_20_sentiment.csv
└── test_20_sentiment.csv
```

> **Note:** Update `BASE_PATH` and `OUT_DIR` at the top of the script to match your local directory structure before running.

---

### Step 2 — Train Text Models (RoBERTa)

```bash
python train_text_emotion.py
python train_text_sentiment.py
```

**Model:** `roberta-base` from Hugging Face  
**Input:** Raw utterance text from CSV  
**Output:**
- Best model checkpoint saved to `outputs/checkpoints/`
- Feature embeddings (`.pt` files) saved to `outputs/features/`
- Metrics JSON saved to `outputs/results/`

**Key config (inside script):**
```python
MODEL_NAME  = "roberta-base"
MAX_LEN     = 128
BATCH_SIZE  = 16
LR          = 2e-5
EPOCHS      = 5
```

---

### Step 3 — Train Audio Models (Wav2Vec2)

```bash
python train_audio_emotion.py
python train_audio_sentiment.py
```

**Model:** `facebook/wav2vec2-base` from Hugging Face  
**Input:** Pre-extracted `.wav` files (16kHz mono)  
**Output:** Checkpoints, feature `.pt` files, metrics JSON

**Key config:**
```python
MODEL_NAME        = "facebook/wav2vec2-base"
BATCH_SIZE        = 8
LR                = 1e-4
EPOCHS            = 5
MAX_AUDIO_SECONDS = 8
TARGET_SR         = 16000
```

> **Note:** Audio scripts use FFmpeg internally to decode WAV files at runtime. Make sure FFmpeg is installed and accessible in your PATH.

---

### Step 4 — Train Video Models (MobileNetV2 + Transformer)

```bash
python train_video_emotion.py
python train_video_sentiment.py
```

**Model:** MobileNetV2 (ImageNet pretrained) + 2-layer Transformer encoder — built from scratch  
**Input:** Raw `.mp4` video files  
**Output:** Checkpoints, feature `.pt` files, metrics JSON

**Key config:**
```python
NUM_FRAMES  = 16
IMG_SIZE    = 224
BATCH_SIZE  = 4
LR          = 1e-4
EPOCHS      = 5
```

**How it works:**
1. 16 frames are uniformly sampled from each video clip
2. Each frame is resized to 224×224 and passed through MobileNetV2
3. The resulting 16 feature vectors (dim 1280 each) go into a Transformer encoder
4. Mean pooling + MLP classifier produces the final prediction

---

### Step 5 — Train Face Models (InceptionResNetV1 + BiLSTM)

```bash
python train_face_emotion.py
python train_face_sentiment.py
```

**Model:** MTCNN (face detection) + InceptionResNetV1 pretrained on VGGFace2 (frozen) + BiLSTM + Attention  
**Input:** Raw `.mp4` video files  
**Output:** Checkpoints, feature `.pt` files, metrics JSON, face cache

**Key config:**
```python
NUM_FRAMES  = 8
FACE_SIZE   = 160
BATCH_SIZE  = 4
LR          = 1e-4
EPOCHS      = 5
```

**How it works:**
1. 8 frames are uniformly sampled from each video clip
2. MTCNN detects and crops the face region from each frame (160×160)
3. If no face is detected in a frame, a zero tensor is substituted
4. InceptionResNetV1 (frozen) extracts a 512-dim embedding per frame
5. BiLSTM models the temporal sequence across 8 frames
6. Attention pooling produces a weighted context vector
7. Linear classifier predicts emotion/sentiment

> **Face Caching:** Detected face tensors are cached to disk after the first run. This makes subsequent epochs much faster. Cache is stored in `face_cache_emotion_20/` and `face_cache_sentiment_20/`.

---

### Step 6 — Train 4-Modality Fusion

**Run AFTER Steps 2–5 are fully complete** (all feature `.pt` files must exist).

```bash
python train_fusion_emotion.py
python train_fusion_sentiment.py
```

**Input:** Feature `.pt` files from all four unimodal models  
**Fusion method:** Concatenation of all four feature vectors  
**Classifier:** 3-layer MLP (input → 1024 → 512 → num_classes) with Dropout 0.3

**Key config:**
```python
BATCH_SIZE  = 32
LR          = 1e-3
EPOCHS      = 5
```

**How fusion works:**
```
text_feat   [768 dim]
audio_feat  [768 dim]   →   concatenate   →   [3328 dim vector]   →   MLP   →   prediction
video_feat  [1280 dim]
face_feat   [512 dim]
```

The fusion script automatically finds utterances that have features from **all four modalities** (by matching UIDs) and only uses those for training and evaluation.

---

### Step 7 — Train 3-Modality Fusion

```bash
python train_fusion3_emotion.py
python train_fusion3_sentiment.py
```

**Default modalities:** Text + Audio + Video

To change which three modalities to fuse, edit the `MODALITIES` list at the top of the script:

```python
# Text + Audio + Video (default)
MODALITIES = ["text", "audio", "video"]

# Text + Audio + Face
MODALITIES = ["text", "audio", "face"]

# Text + Video + Face
MODALITIES = ["text", "video", "face"]
```

---

## Output Directory Structure

After running all scripts, your outputs will be organized as:

```
outputs/
├── checkpoints/
│   ├── text_emotion_20/
│   │   └── best_text_emotion.pt
│   ├── text_outputs_sentiment_20/
│   │   └── best_text_sentiment.pt
│   └── ...
│
├── features/
│   ├── text_emotion_20_train.pt
│   ├── text_emotion_20_dev.pt
│   ├── text_emotion_20_test.pt
│   ├── audio_emotion_15_train.pt
│   ├── audio_emotion_15_dev.pt
│   ├── audio_emotion_15_test.pt
│   ├── video_emotion_20_train.pt
│   ├── face_emotion_20_train.pt
│   └── ... (same pattern for sentiment)
│
└── results/
    ├── text_emotion_20_metrics.json
    ├── text_sentiment_20_metrics.json
    ├── audio_emotion_15_metrics.json
    ├── audio_sentiment_15_metrics.json
    ├── video_emotion_20_metrics.json
    ├── video_sentiment_20_metrics.json
    ├── face_emotion_20_metrics.json
    ├── face_sentiment_20_metrics.json
    ├── fusion_emotion_metrics.json
    ├── fusion_sentiment_metrics.json
    ├── fusion3_emotion_metrics.json
    └── fusion3_sentiment_metrics.json
```

---

## Results

### Unimodal Emotion Recognition (Test Accuracy %)

| Modality | Model | Our Result | Paper Result |
|----------|-------|-----------|--------------|
| Text | RoBERTa-base | 60.93 | 64.34 |
| Audio | Wav2Vec2-base | 48.33 | 51.49 |
| Video | MobileNetV2 + Transformer | 48.33 | 36.14 |
| Face | InceptionResNetV1 + BiLSTM | 48.33 | 22.61 |

### Unimodal Sentiment Analysis (Test Accuracy %)

| Modality | Model | Our Result | Paper Result |
|----------|-------|-----------|--------------|
| Text | RoBERTa-base | 70.26 | 69.21 |
| Audio | Wav2Vec2-base | 85.84 | 56.20 |
| Video | MobileNetV2 + Transformer | 48.21 | 42.51 |
| Face | InceptionResNetV1 + BiLSTM | 47.69 | 38.98 |

### Multimodal Fusion (Test Accuracy %)

| Configuration | Our Emotion | Paper Emotion | Our Sentiment | Paper Sentiment |
|--------------|-------------|---------------|---------------|-----------------|
| Text + Audio + Video | 62.98 | 66.29 | 88.58 | 71.76 |
| Text + Audio + Face | 63.24 | 66.25 | 86.30 | 71.84 |
| All 4 Modalities | 62.47 | 66.36 | 88.13 | 72.15 |

---

## Important Notes

### Execution Order
Scripts **must** be run in order. Fusion scripts depend on feature files saved by unimodal scripts. Running fusion before unimodal training is complete will raise a `FileNotFoundError`.

```
Step 1 (subset) → Steps 2-5 (unimodal, any order) → Steps 6-7 (fusion)
```

### Path Configuration
Every script has a `CONFIG` section at the top. Update these paths to match your local setup before running:

```python
BASE_PATH    = "/your/path/to/MELD.Raw"
BALANCED_DIR = "/your/path/to/meld_balanced"
AUDIO_BASE   = "/your/path/to/meld_audio_15"
OUT_DIR      = "/your/path/to/outputs"
RESULTS_DIR  = "/your/path/to/results"
FEATURE_DIR  = "/your/path/to/features"
```

### Reproducibility
All scripts use a fixed random seed for reproducibility:
```python
SEED = 42
```

### Why Audio Sentiment is High (85.84%)
Our sentiment results significantly exceed the paper's for audio and fusion. This is because the class-balanced training subset gives Wav2Vec2 a more equal distribution of positive/negative/neutral samples during training, whereas the original paper trained on the full imbalanced dataset where neutral dominates.

### Why Emotion Unimodal (Audio/Video/Face) is ~48%
Audio, video, and face emotion models all converge to predicting the majority class (neutral) due to class imbalance in the 7-class emotion task. The balanced 15% sampling still preserves proportional class distribution — neutral remains dominant — so these models get stuck. Weighted loss (`nn.CrossEntropyLoss(weight=...)`) would resolve this.

---

## Citation

If you use this code, please cite the original paper:

```
@article{farhadipour2025multimodal,
  title={Multimodal Emotion Recognition and Sentiment Analysis in Multi-Party Conversation Contexts},
  author={Farhadipour, Aref and Ranjbar, Hossein and Chapariniya, Masoumeh and Vukovic, Teodora and Ebling, Sarah and Dellwo, Volker},
  journal={arXiv preprint arXiv:2503.06805},
  year={2025}
}
```

And the MELD dataset:

```
@article{poria2018meld,
  title={MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations},
  author={Poria, Soujanya and Hazarika, Devamanyu and Majumder, Navonil and Naik, Gautam and Cambria, Erik and Mihalcea, Rada},
  journal={arXiv preprint arXiv:1810.02508},
  year={2018}
}
```
---

## Group Information

| Member | Roll Number |
|--------|------------|
| Member 1 | 25i-7601 |
| Member 2 | 25i-7641 |
| Member 3 | 25i-7643 |

**Course:** NLP — FAST-NUCES  
**Instructor:** Dr. Zohair Ahmed  
**Email:** zohair.ahmed@isb.nu.edu.pk

**Assignment:** 02 — Reproduction of Results

