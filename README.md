# Speech-based Grammar Score Prediction 🎤📊

> **Competition**: SHL-Internship Assessment  
> **Task**: Predict continuous grammar scores (1-5) from spoken audio samples  
> **Evaluation Metrics**: Pearson Correlation & RMSE

---

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Solution Approach](#solution-approach)
- [Architecture](#architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Training Details](#training-details)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements a **multimodal deep learning solution** for automated grammar scoring of spoken English audio samples. The system combines acoustic features from audio and linguistic features from text transcripts to predict continuous grammar scores ranging from 1 (poor) to 5 (excellent).

### Key Features

✅ **Multimodal Architecture** - Fuses audio (WavLM) and text (BERT) features  
✅ **Advanced Fusion** - Bidirectional cross-attention mechanism  
✅ **Robust Training** - 5-fold cross-validation with ensemble  
✅ **State-of-the-art Techniques** - SWA, mixed precision, advanced augmentation  
✅ **Production-Ready** - Complete preprocessing and inference pipeline  

---

## 📝 Problem Statement

### Grammar Score Rubric

| Score | Description |
|-------|-------------|
| **1** | Limited control over sentence structure and syntax; struggles with basic grammatical structures |
| **2** | Limited understanding with consistent basic mistakes; may leave sentences incomplete |
| **3** | Decent grasp of sentence structure with grammatical errors, or vice versa |
| **4** | Strong understanding with good control; occasional minor errors that don't cause misunderstandings |
| **5** | High grammatical accuracy with adept control of complex structures; seldom makes noticeable mistakes |

### Dataset

- **Training**: 409 audio samples (45-60 seconds each)
- **Testing**: 197 audio samples
- **Format**: WAV audio files + CSV transcripts
- **Labels**: Continuous MOS Likert Grammar Scores (1-5)

---

## 🚀 Solution Approach

### 1. Multimodal Architecture

Our solution leverages **two complementary modalities**:

#### 🎵 Audio Branch
- **Model**: Microsoft WavLM-base (94M parameters)
- **Purpose**: Captures prosody, fluency, speech patterns, and acoustic cues
- **Features**: Multi-head attention pooling over temporal features

#### 📝 Text Branch
- **Model**: BERT-base-uncased (110M parameters)
- **Purpose**: Analyzes grammar, syntax, and linguistic structure
- **Features**: [CLS] token representation with contextual embeddings

#### 🔀 Fusion Layer
- **Mechanism**: Bidirectional cross-attention (8 heads)
- **Innovation**: Gated fusion learns optimal combination weights
- **Output**: 768-dimensional multimodal representation

#### 🎯 Regression Head
- **Architecture**: Deep 6-layer MLP (768→512→384→256→128→64→1)
- **Regularization**: Layer normalization, GELU activation, dropout
- **Output**: Continuous score scaled to [1, 5]

### 2. Training Strategy

```
┌─────────────────────────────────────────────────────────┐
│  5-Fold Cross-Validation                                │
├─────────────────────────────────────────────────────────┤
│  • Stratified splits for balanced distribution          │
│  • 25 epochs per fold with early stopping               │
│  • Weighted ensemble by Pearson correlation             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Advanced Optimization                                  │
├─────────────────────────────────────────────────────────┤
│  • AdamW optimizer (lr=3e-5, wd=0.02)                   │
│  • OneCycleLR scheduler with warmup                     │
│  • Gradient accumulation (effective batch size: 12)     │
│  • Mixed precision training (FP16)                      │
│  • Gradient clipping (max norm: 1.0)                    │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Regularization Techniques                              │
├─────────────────────────────────────────────────────────┤
│  • Stochastic Weight Averaging (from epoch 12)          │
│  • Layer freezing (first 4 layers of encoders)          │
│  • Dropout (0.1-0.3 throughout network)                 │
│  • Early stopping (patience: 8, min_delta: 0.0001)      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  Data Augmentation                                      │
├─────────────────────────────────────────────────────────┤
│  • Time stretching (0.9-1.1x)                           │
│  • Pitch shifting (±5%)                                 │
│  • Gaussian noise injection                             │
│  • Time shifting (±15%)                                 │
│  • Volume perturbation (0.8-1.2x)                       │
└─────────────────────────────────────────────────────────┘
```

### 3. Custom Loss Function

**RMSE-Focused Loss** = 0.5 × MSE + 0.3 × Huber + 0.2 × Ordinal

- **MSE**: Direct RMSE optimization
- **Huber**: Robustness to outliers (δ=0.5)
- **Ordinal**: Encourages correct score ordering

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                               │
├──────────────────────────────────────────────────────────────────┤
│  Audio: (batch, 160000)     Text: (batch, 512)                   │
└──────────────────────────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────┴─────────────────────┐
        │                                           │
┌───────▼─────────┐                       ┌────────▼────────┐
│  WavLM-base     │                       │   BERT-base     │
│  (Frozen: 0-3)  │                       │  (Frozen: 0-3)  │
│  768-dim output │                       │  768-dim output │
└───────┬─────────┘                       └────────┬────────┘
        │                                          │
        │ Multi-head                               │ [CLS]
        │ Attention Pool                           │ Token
        │ (8 heads)                                │
        │                                          │
┌───────▼─────────┐                       ┌────────▼────────┐
│  Projection     │                       │  Projection     │
│  768→512→384    │                       │  768→512→384    │
│  [LN+GELU+Drop] │                       │  [LN+GELU+Drop] │
└───────┬─────────┘                       └────────┬────────┘
        │                                          │
        └─────────────────────┬────────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │    Bidirectional Cross-Attention        │
        │    (8 heads, audio↔text)                │
        │    + Residual Connections               │
        └─────────────────────┬───────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │         Gated Fusion Module             │
        │    (Learnable combination weights)      │
        └─────────────────────┬───────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │         Regression Head (768-dim)       │
        │  768→512→384→256→128→64→1               │
        │  [Each: Linear+LN+GELU+Dropout]         │
        └─────────────────────┬───────────────────┘
                              ↓
        ┌─────────────────────────────────────────┐
        │    OUTPUT: Grammar Score [1.0, 5.0]     │
        └─────────────────────────────────────────┘

Total Parameters: ~210M
Trainable Parameters: ~25M (12%)
```

---

## 📊 Results

### Cross-Validation Performance

| Fold | Pearson Correlation | RMSE |
|------|---------------------|------|
| 1    | 0.6484             | 0.6393 |
| 2    | 0.5674             | 0.6661 |
| 3    | 0.8306             | 0.5043 |
| 4    | 0.5603             | 0.6151 |
| 5    | 0.7714             | 0.4997 |
| **Mean** | **0.6756 ± 0.1121** | **0.5849 ± 0.0726** |

### Overall Metrics

- **Overall CV Pearson**: 0.6756
- **Overall CV RMSE**: 0.5849
- **Training Time**: ~2.5 hours on single GPU (T4)

### Score Distribution

```
Training Data Distribution:
  Mean: 3.42 | Std: 0.89 | Range: [1.0, 5.0]

Test Predictions Distribution:
  Mean: 3.38 | Std: 0.76 | Range: [1.2, 4.8]
```

---

## 🔧 Installation

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 8GB+ GPU memory (recommended)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd grammar-score-prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets librosa soundfile scikit-learn pandas numpy matplotlib seaborn tqdm
```

4. **Download pre-trained models** (optional, will auto-download on first run)
```bash
# Models will be downloaded from HuggingFace Hub:
# - microsoft/wavlm-base
# - bert-base-uncased
```

---

## 💻 Usage

### Training

1. **Update configuration in `train_grammar_model.py`**:

```python
CONFIG = {
    "train_audio_dir": "/path/to/train/audios",
    "test_audio_dir": "/path/to/test/audios",
    "train_csv": "/path/to/train.csv",
    "test_csv": "/path/to/test.csv",
    "train_transcripts_csv": "/path/to/train_transcripts.csv",
    "test_transcripts_csv": "/path/to/test_transcripts.csv",
    # ... other settings
}
```

2. **Run training**:

```bash
python train_grammar_model.py
```

**Output**:
- Trained model checkpoints: `model_fold_1.pt` through `model_fold_5.pt`
- Training plots: `output/multimodal_fold_*.png`
- CV summary: `output/cv_summary.png`

### Inference

Use the submission notebook `grammarscoreengine.ipynb` for generating predictions:

1. Open the notebook in Jupyter/Kaggle
2. Update model paths in the CONFIG section
3. Run all cells
4. Output: `submission.csv` with predictions

---

## 📁 Project Structure

```
grammar-score-prediction/
│
├── train_grammar_model.py      # Training script (converted from notebook)
├── grammarscoreengine.ipynb    # Submission notebook with inference
├── README.md                    # This file
│
├── output/                      # Training outputs (auto-created)
│   ├── model_fold_1.pt
│   ├── model_fold_2.pt
│   ├── model_fold_3.pt
│   ├── model_fold_4.pt
│   ├── model_fold_5.pt
│   ├── cv_summary.png
│   ├── multimodal_fold_1.png
│   ├── multimodal_fold_2.png
│   └── ...
│
├── data/                        # Dataset (not included in repo)
│   ├── audios/
│   │   ├── train/
│   │   └── test/
│   └── csvs/
│       ├── train.csv
│       ├── test.csv
│       ├── train_transcripts.csv
│       └── test_transcripts.csv
│
└── submission.csv               # Final predictions
```

---

## 🧠 Model Details

### Audio Processing

**Input**: WAV files (45-60 seconds, various sample rates)

**Preprocessing**:
1. Resample to 16kHz
2. Convert to mono
3. Peak normalization to [-1, 1]
4. Pad/crop to 10 seconds (160,000 samples)

**Feature Extraction**:
- WavLM-base encoder extracts contextualized audio representations
- Multi-head attention pooling aggregates temporal information
- Output: 768-dimensional audio embedding

### Text Processing

**Input**: Speech transcripts (from ASR system)

**Preprocessing**:
1. WordPiece tokenization (BERT tokenizer)
2. Truncate/pad to 512 tokens
3. Add special tokens: [CLS] ... [SEP]

**Feature Extraction**:
- BERT-base encoder produces contextual token embeddings
- Extract [CLS] token representation
- Output: 768-dimensional text embedding

### Fusion Mechanism

**Cross-Attention**:
```
Audio → Query | Text → Key, Value  →  Attended Audio
Text  → Query | Audio → Key, Value →  Attended Text
```

**Gated Fusion**:
```python
gate = σ(MLP([audio; text]))
fused = gate ⊙ audio + (1-gate) ⊙ text
```

This allows the model to learn which modality is more informative for each sample.

---

## 🎓 Training Details

### Hyperparameters

| Parameter         | Value                               |
|-------------------|-------------------------------------|
| Batch Size        | 6 (effective: 12 with accumulation) |
| Learning Rate     | 3e-5                                |
| Weight Decay      | 0.02                                |
| Optimizer         | AdamW (β1=0.9, β2=0.999)            |
| Scheduler         | OneCycleLR (warmup: 15%)            |
| Gradient Clipping | 1.0                                 |
| Epochs            | 25 (with early stopping)            |
| Mixed Precision   | FP16                                |
| SWA Start         | Epoch 12                            |

### Data Augmentation

Applied with 60% probability during training:

- **Time Stretching**: 0.9-1.1x (40% of augmented samples)
- **Pitch Shifting**: ±5% (30%)
- **Gaussian Noise**: 0.001-0.008 level (50%)
- **Time Shifting**: ±15% (40%)
- **Volume Perturbation**: 0.8-1.2x (50%)

### Computational Requirements

- **Training**: ~2.5 hours on NVIDIA T4 GPU
- **Memory**: 8GB GPU, 16GB RAM
- **Storage**: ~2GB for model checkpoints

---

## 🔮 Inference

### Ensemble Prediction

The final submission uses a **weighted ensemble** of all 5 fold models:

```python
# Weights based on fold Pearson correlations
weights = [0.6484, 0.5674, 0.8306, 0.5603, 0.7714]
weights = weights / sum(weights)  # Normalize

# Ensemble prediction
pred = sum(w * model_i(x) for w, model_i in zip(weights, models))
pred = clip(pred, 1.0, 5.0)
```

### Test-Time Augmentation (Optional)

For even more robust predictions:

```python
# Generate multiple augmented versions
preds = [model(augment(x)) for _ in range(3)]
final_pred = mean(preds)
```

---

## 📈 Evaluation

### Metrics

1. **Pearson Correlation** (Primary)
   - Measures linear relationship between predictions and ground truth
   - Range: [-1, 1], higher is better

2. **RMSE** (Root Mean Squared Error)
   - Measures prediction accuracy
   - Range: [0, ∞), lower is better

### Validation Strategy

- **5-Fold Cross-Validation**: Ensures robust generalization
- **Stratified Splits**: Maintains label distribution across folds
- **Out-of-Fold Predictions**: Used for ensemble weight calibration

---

## 🚀 Future Improvements

### Architecture Enhancements

1. **Hierarchical Attention**: Better handling of long audio sequences
2. **Multi-Task Learning**: Joint prediction of fluency, pronunciation, vocabulary
3. **Uncertainty Quantification**: Bayesian approaches for confidence estimation
4. **Transformer XL**: Longer context modeling

### Training Improvements

1. **Curriculum Learning**: Train on easy samples first
2. **Focal Loss**: Focus on hard-to-classify samples
3. **Active Learning**: Select most informative samples for labeling
4. **Knowledge Distillation**: Compress ensemble into single model

### Data Enhancements

1. **Synthetic Data**: Generate augmented samples with TTS
2. **Back-Translation**: Text augmentation for better linguistic diversity
3. **External Data**: Leverage larger speech/grammar datasets
4. **Semi-Supervised Learning**: Use unlabeled audio samples

### Deployment

1. **Model Quantization**: INT8 for faster inference
2. **ONNX Export**: Framework-agnostic deployment
3. **REST API**: Web service for real-time scoring
4. **Mobile Optimization**: TensorFlow Lite for on-device inference

---

## 🎯 Key Takeaways

### What Worked Well

✅ **Multimodal Fusion**: Combining audio and text provides complementary information  
✅ **Cross-Attention**: Allows modalities to interact and inform each other  
✅ **SWA**: Improves generalization by averaging model weights  
✅ **Mixed Precision**: Speeds up training without sacrificing quality  
✅ **Ensemble**: 5-fold ensemble significantly reduces variance  

### Challenges Faced

⚠️ **Small Dataset**: 409 training samples limits model capacity  
⚠️ **Label Subjectivity**: Grammar scoring has inherent annotation variance  
⚠️ **Transcript Quality**: ASR errors can impact text branch performance  
⚠️ **Computational Cost**: Large models require significant GPU resources  

### Lessons Learned

💡 **Less is More**: Freezing early layers prevents overfitting on small data  
💡 **Augmentation Matters**: Audio augmentation crucial for generalization  
💡 **Loss Design**: Custom loss function tailored to RMSE metric improves results  
💡 **Careful Tuning**: Learning rate and scheduler critical for convergence  

---

## 📚 References

### Models

- **WavLM**: [WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)
- **BERT**: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

### Techniques

- **SWA**: [Averaging Weights Leads to Wider Optima and Better Generalization](https://arxiv.org/abs/1803.05407)
- **Mixed Precision**: [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- **Cross-Attention**: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

---

## 👤 Author

**Aryan Verma**
- GitHub: [@Aryan-Verma-999](https://github.com/Aryan-Verma-999)
- Email: aryan-999@outlook.com