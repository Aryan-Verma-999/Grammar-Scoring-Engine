# Grammar Scoring Engine

> **Automated Grammar Scoring System for Spoken English Audio Samples**  
> *SHL Internship Assessment Submission*

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Evaluation Metrics:** Pearson Correlation & RMSE  
**Task:** Predict continuous grammar scores (1-5) from 45-60 second audio samples

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technical Details](#technical-details)
- [Future Improvements](#future-improvements)

---

## Overview

This project implements a **multimodal deep learning solution** that automatically scores the grammatical quality of spoken English. The system processes audio files and predicts grammar scores on a 1-5 scale, mimicking human expert evaluation.

### Key Features

- **Automatic Speech Transcription:** Uses Whisper-large-v3 for high-accuracy audio-to-text conversion
- **Multimodal Learning:** Combines audio (WavLM) and text (BERT) features
- **Bidirectional Fusion:** Cross-attention mechanism for audio-text interaction
- **Robust Training:** 5-fold cross-validation with ensemble predictions
- **Production-Ready:** Complete preprocessing and inference pipeline

---

## Problem Statement

### Dataset
- **Training:** 409 audio samples (45-60 seconds each)
- **Testing:** 197 audio samples
- **Format:** WAV audio files with continuous MOS Likert Grammar Scores

### Grammar Score Rubric

| Score | Description |
|-------|-------------|
| **1** | Limited control over sentence structure and syntax; struggles with basic grammatical structures |
| **2** | Limited understanding with consistent basic mistakes; may leave sentences incomplete |
| **3** | Decent grasp of sentence structure with grammatical errors, or vice versa |
| **4** | Strong understanding with good control; occasional minor errors that don't cause misunderstandings |
| **5** | High grammatical accuracy with adept control of complex structures; seldom makes noticeable mistakes |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     AUDIO INPUT (WAV)                       │
│                    (45-60 seconds)                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 1: TRANSCRIPTION                          │
│         transcription_script.py (Whisper-large-v3)          │
│                                                             │
│  • Loads audio files from directory                         │
│  • Uses OpenAI Whisper-large-v3 model                       │
│  • Generates text transcripts                               │
│  • Saves to train_transcripts.csv / test_transcripts.csv    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 2: MODEL TRAINING                         │
│            train_grammar_model.py                           │
│                                                             │
│  ┌─────────────────┐         ┌──────────────────┐           │
│  │  Audio Stream   │         │  Text Stream     │           │
│  │  (WavLM-base)   │         │  (BERT-base)     │           │
│  │                 │         │                  │           │
│  │  • Resample     │         │  • Tokenize      │           │
│  │  • Normalize    │         │  • Pad/Truncate  │           │
│  │  • Extract      │         │  • Encode        │           │
│  │    Features     │         │    Text          │           │
│  │  • 768-dim      │         │  • 768-dim       │           │
│  └────────┬────────┘         └────────┬─────────┘           │
│           │                           │                     │
│           └───────────┬───────────────┘                     │
│                       ▼                                     │
│           ┌───────────────────────┐                         │
│           │  Cross-Attention      │                         │
│           │  Fusion Module        │                         │
│           │  (Bidirectional)      │                         │
│           └───────────┬───────────┘                         │
│                       ▼                                     │
│           ┌───────────────────────┐                         │
│           │  Regression Head      │                         │
│           │  (6-layer MLP)        │                         │
│           └───────────┬───────────┘                         │
│                       ▼                                     │
│              Grammar Score [1-5]                            │
│                                                             │
│  • 5-fold cross-validation                                  │
│  • Advanced optimization (AdamW + OneCycleLR)               │
│  • Saves 5 model checkpoints                                │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              STEP 3: INFERENCE                              │
│          grammarscoreengine.ipynb                           │
│                                                             │
│  • Loads all 5 trained models                               │
│  • Processes test audio files                               │
│  • Generates ensemble predictions                           │
│  • Creates submission.csv                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## How It Works

### Phase 1: Audio Transcription

**Script:** `transcription_script.py`

This script converts audio files to text transcripts using OpenAI's Whisper-large-v3 model.

```python
# Key functionality:
1. Load audio files from train/test directories
2. Initialize Whisper-large-v3 model (1550M parameters)
3. Transcribe each audio file to text
4. Save results to CSV files:
   - train_transcripts.csv (409 samples)
   - test_transcripts.csv (197 samples)
```

**Why Whisper-large-v3?**
- State-of-the-art speech recognition accuracy
- Robust to accents, background noise, and speech variations
- Provides high-quality text for downstream grammar analysis

**Output Format:**
```csv
filename,transcript
audio_001.wav,"Hello my name is John and I am here to..."
audio_002.wav,"Today I would like to discuss about..."
```

---

### Phase 2: Model Training

**Script:** `train_grammar_model.py`

This script trains the multimodal grammar scoring model using both audio and text features.

#### 2.1 Audio Processing Pipeline

```python
# WavLM-base encoder (94M parameters)
1. Load audio file (WAV format)
2. Resample to 16kHz (standard sample rate)
3. Convert to mono (single channel)
4. Normalize amplitude to [-1, 1]
5. Pad or crop to 10 seconds (160,000 samples)
6. Extract features using WavLM encoder
7. Apply multi-head attention pooling (8 heads)
8. Output: 768-dimensional audio embedding
```

**Audio Features Captured:**
- Prosody (rhythm, stress, intonation)
- Fluency (pauses, hesitations)
- Speech rate and tempo
- Acoustic patterns indicating grammar quality

#### 2.2 Text Processing Pipeline

```python
# BERT-base-uncased encoder (110M parameters)
1. Load transcript from CSV
2. Tokenize using WordPiece tokenizer
3. Add special tokens: [CLS] ... [SEP]
4. Truncate or pad to 512 tokens
5. Convert to input IDs and attention masks
6. Extract features using BERT encoder
7. Use [CLS] token representation
8. Output: 768-dimensional text embedding
```

**Linguistic Features Captured:**
- Grammatical structures and syntax
- Sentence complexity
- Word usage and vocabulary
- Contextual relationships

#### 2.3 Multimodal Fusion

```python
# Bidirectional Cross-Attention Mechanism
1. Audio → Query | Text → Key, Value → Attended Audio
2. Text → Query | Audio → Key, Value → Attended Text
3. Gated Fusion Module:
   gate = sigmoid(MLP([audio_features; text_features]))
   fused = gate ⊙ attended_audio + (1-gate) ⊙ attended_text
4. Learned combination: Model decides which modality is more important
```

#### 2.4 Training Strategy

```python
# 5-Fold Cross-Validation
for fold in range(1, 6):
    1. Split data stratified by score distribution
    2. Initialize model with fresh weights
    3. Train for 25 epochs with early stopping
    4. Apply data augmentation (60% probability):
       - Time stretching: 0.9-1.1x
       - Pitch shifting: ±5%
       - Gaussian noise injection
       - Time shifting: ±15%
       - Volume perturbation: 0.8-1.2x
    5. Use mixed precision training (FP16)
    6. Apply Stochastic Weight Averaging (SWA) from epoch 12
    7. Save best model checkpoint
    8. Evaluate on validation fold

# After all folds: weighted ensemble based on Pearson correlation
```

**Training Outputs:**
- `model_fold_1.pt` through `model_fold_5.pt` (model checkpoints)
- Training plots showing loss curves and metrics
- Cross-validation summary with performance statistics

---

### Phase 3: Inference & Submission

**Notebook:** `grammarscoreengine.ipynb`

```python
# Inference Pipeline
1. Load all 5 trained model checkpoints
2. For each test audio file:
   a. Transcribe using Whisper (if not already done)
   b. Process audio through WavLM
   c. Process text through BERT
   d. Get predictions from all 5 models
   e. Compute weighted ensemble prediction
   f. Clip score to [1.0, 5.0] range
3. Save predictions to submission.csv
```

**Ensemble Strategy:**
```python
# Weights based on fold validation Pearson correlations
weights = [0.6484, 0.5674, 0.8306, 0.5603, 0.7714]
weights = normalize(weights)

# Final prediction
final_score = Σ(weight_i × model_i_prediction)
final_score = clip(final_score, 1.0, 5.0)
```

---

## Installation

### Prerequisites

- Python 3.11 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 8GB+ GPU memory (recommended)

### Step 1: Clone Repository

```bash
git clone https://github.com/Aryan-Verma-999/Grammar-Scoring-Engine.git
cd Grammar-Scoring-Engine
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install required libraries
pip install transformers datasets librosa soundfile scikit-learn pandas numpy matplotlib seaborn tqdm

# For Whisper transcription
pip install openai-whisper
```

### Step 4: Download Models (Optional)

Models will auto-download on first run, but you can pre-download:

```python
from transformers import AutoModel, AutoTokenizer
import whisper

# Download WavLM
AutoModel.from_pretrained("microsoft/wavlm-base")

# Download BERT
AutoModel.from_pretrained("bert-base-uncased")
AutoTokenizer.from_pretrained("bert-base-uncased")

# Download Whisper
whisper.load_model("large-v3")
```

---

## Usage Guide

### Quick Start (3 Steps)

#### **Step 1: Transcribe Audio Files**

```bash
python transcription_script.py
```

**Configuration:**
```python
# Edit paths in transcription_script.py
TRAIN_AUDIO_DIR = "/path/to/train/audios"
TEST_AUDIO_DIR = "/path/to/test/audios"
OUTPUT_DIR = "/path/to/save/transcripts"
```

**Output:**
- `train_transcripts.csv` - Training audio transcriptions
- `test_transcripts.csv` - Test audio transcriptions

**Runtime:** ~1-2 hours for 409+197 audio files on GPU

---

#### **Step 2: Train Grammar Model**

```bash
python train_grammar_model.py
```

**Configuration:**
```python
# Edit CONFIG dictionary in train_grammar_model.py
CONFIG = {
    "train_audio_dir": "/path/to/train/audios",
    "test_audio_dir": "/path/to/test/audios",
    "train_csv": "/path/to/train.csv",  # Contains filenames and labels
    "test_csv": "/path/to/test.csv",
    "train_transcripts_csv": "/path/to/train_transcripts.csv",
    "test_transcripts_csv": "/path/to/test_transcripts.csv",
    "output_dir": "./output",
    "batch_size": 6,
    "num_epochs": 25,
    "learning_rate": 3e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}
```

**What Happens:**
1. Loads audio files and transcripts
2. Trains 5 models with cross-validation
3. Saves checkpoints after each fold
4. Generates training plots and metrics
5. Computes overall RMSE and Pearson correlation

**Output:**
- `output/model_fold_1.pt` to `model_fold_5.pt` (trained models)
- `output/multimodal_fold_*.png` (training curves)
- `output/cv_summary.png` (cross-validation results)
- Console output with RMSE and correlation scores

**Runtime:** ~2.5 hours on NVIDIA T4 GPU

---

#### **Step 3: Generate Predictions**

```bash
# Open Jupyter notebook
jupyter notebook grammarscoreengine.ipynb
```

**In the Notebook:**
1. Update model paths in CONFIG section
2. Run all cells sequentially
3. Final cell generates `submission.csv`

**Output Format:**
```csv
filename,label
test_001.wav,3.45
test_002.wav,4.12
test_003.wav,2.89
...
```

---

## Project Structure

```
Grammar-Scoring-Engine/
│
├── transcription_script.py          # STEP 1: Audio → Text using Whisper-large-v3
├── train_grammar_model.py           # STEP 2: Train multimodal model
├── grammarscoreengine.ipynb         # STEP 3: Inference & submission
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
│
├── data/                            # Dataset (not included)
│   ├── audios/
│   │   ├── train/                   # 409 training audio files
│   │   └── test/                    # 197 test audio files
│   └── csvs/
│       ├── train.csv                # Training labels
│       ├── test.csv                 # Test filenames
│       ├── train_transcripts.csv    # Generated by transcription_script.py
│       └── test_transcripts.csv     # Generated by transcription_script.py
│
├── output/                          # Training outputs (auto-created)
│   ├── model_fold_1.pt             # Trained model checkpoint (fold 1)
│   ├── model_fold_2.pt             # Trained model checkpoint (fold 2)
│   ├── model_fold_3.pt             # Trained model checkpoint (fold 3)
│   ├── model_fold_4.pt             # Trained model checkpoint (fold 4)
│   ├── model_fold_5.pt             # Trained model checkpoint (fold 5)
│   ├── cv_summary.png              # Cross-validation results plot
│   └── multimodal_fold_*.png       # Training curves for each fold
│
└── submission.csv                   # Final predictions (generated by notebook)
```

---

## Model Performance

### Cross-Validation Results

| Fold | Pearson Correlation | RMSE   |
|------|---------------------|--------|
| 1    | 0.6484              | 0.6393 |
| 2    | 0.5674              | 0.6661 |
| 3    | 0.8306              | 0.5043 |
| 4    | 0.5603              | 0.6151 |
| 5    | 0.7714              | 0.4997 |
| **Mean** | **0.6756 ± 0.1121** | **0.5849 ± 0.0726** |

### Training Metrics

- **Overall CV Pearson Correlation:** 0.6756
- **Overall CV RMSE:** 0.5849 (Required for submission)
- **Training Time:** ~2.5 hours on NVIDIA T4 GPU
- **Model Size:** ~210M total parameters, ~25M trainable (12%)

### Data Distribution

**Training Data:**
- Mean: 3.42 | Std: 0.89 | Range: [1.0, 5.0]

**Test Predictions:**
- Mean: 3.38 | Std: 0.76 | Range: [1.2, 4.8]

---

## Technical Details

### Model Architecture

#### Audio Encoder (WavLM-base)
```python
Input: Audio waveform (160,000 samples @ 16kHz)
Encoder: 12 transformer layers (94M parameters)
Pooling: Multi-head attention (8 heads)
Output: 768-dimensional feature vector
```

#### Text Encoder (BERT-base)
```python
Input: Tokenized transcript (max 512 tokens)
Encoder: 12 transformer layers (110M parameters)
Extraction: [CLS] token representation
Output: 768-dimensional feature vector
```

#### Fusion Module
```python
Cross-Attention: Bidirectional (audio↔text, 8 heads)
Gated Fusion: Learnable combination weights
Projection: 768 → 512 → 384 dimensions
```

#### Regression Head
```python
Architecture: 6-layer MLP
Layers: 768 → 512 → 384 → 256 → 128 → 64 → 1
Activation: GELU
Regularization: LayerNorm + Dropout (0.1-0.3)
Output: Single continuous score
```

### Training Configuration

```python
Optimizer: AdamW
  - Learning rate: 3e-5
  - Weight decay: 0.02
  - Betas: (0.9, 0.999)

Scheduler: OneCycleLR
  - Max LR: 3e-5
  - Warmup: 15% of steps
  - Annealing: Cosine

Regularization:
  - Gradient clipping: max_norm=1.0
  - Dropout: 0.1-0.3 throughout
  - Layer freezing: First 4 layers of encoders
  - Early stopping: patience=8, min_delta=0.0001

Advanced Techniques:
  - Mixed precision training (FP16)
  - Gradient accumulation (steps=2)
  - Stochastic Weight Averaging (from epoch 12)
  - Data augmentation (60% probability)
```

### Loss Function

```python
# RMSE-optimized composite loss
loss = 0.5 × MSE + 0.3 × Huber + 0.2 × Ordinal

Where:
- MSE: Mean Squared Error (direct RMSE optimization)
- Huber: Robust to outliers (delta=0.5)
- Ordinal: Encourages correct score ordering
```

### Data Augmentation (Audio)

Applied during training with 60% probability:

```python
Augmentations:
1. Time Stretching: 0.9-1.1x (40% of augmented samples)
2. Pitch Shifting: ±5% (30%)
3. Gaussian Noise: 0.001-0.008 level (50%)
4. Time Shifting: ±15% (40%)
5. Volume Perturbation: 0.8-1.2x (50%)
```

---

## Future Improvements

### Model Enhancements
- **Hierarchical Attention:** Better handling of long audio sequences
- **Multi-Task Learning:** Joint prediction of fluency, pronunciation, vocabulary
- **Uncertainty Quantification:** Bayesian approaches for confidence estimation
- **Transformer-XL:** Longer context modeling for extended audio

### Training Improvements
- **Curriculum Learning:** Progressive training from easy to hard samples
- **Focal Loss:** Focus on hard-to-classify samples
- **Active Learning:** Select most informative samples for labeling
- **Knowledge Distillation:** Compress ensemble into single efficient model

### Data Augmentation
- **Synthetic Data:** Generate augmented samples with TTS
- **Back-Translation:** Text augmentation for linguistic diversity
- **External Data:** Leverage larger speech/grammar datasets
- **Semi-Supervised Learning:** Use unlabeled audio samples

### Deployment Optimization
- **Model Quantization:** INT8 for faster inference
- **ONNX Export:** Framework-agnostic deployment
- **REST API:** Web service for real-time scoring
- **Mobile Optimization:** TensorFlow Lite for on-device inference

---

## Key Learnings

✅ **Multimodal Fusion Works:** Combining audio and text provides complementary information  
✅ **Cross-Attention is Powerful:** Allows modalities to interact and inform each other  
✅ **SWA Improves Generalization:** Weight averaging reduces overfitting  
✅ **Ensemble Reduces Variance:** 5-fold ensemble significantly improves stability  
💡 **Less is More:** Freezing early layers prevents overfitting on small datasets  
💡 **Augmentation Matters:** Audio augmentation crucial for generalization  
💡 **Loss Design is Critical:** Custom loss tailored to RMSE improves results

---

## References

- **WavLM:** [Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
- **BERT:** [Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- **Whisper:** [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)
- **SWA:** [Averaging Weights Leads to Wider Optima](https://arxiv.org/abs/1803.05407)
- **Cross-Attention:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## Author

**Aryan Verma**
- GitHub: [@Aryan-Verma-999](https://github.com/Aryan-Verma-999)
- Email: aryan-999@outlook.com

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- SHL for providing the assessment opportunity
- HuggingFace for pre-trained models
- OpenAI for Whisper model

---

**If you find this project helpful, please consider giving it a star!**
