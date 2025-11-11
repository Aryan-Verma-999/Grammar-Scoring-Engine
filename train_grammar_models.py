"""
Speech-based Grammar Score Prediction - Training Script
========================================================
This script trains a multimodal (audio + text) deep learning model
for predicting grammar scores from spoken audio samples.

Competition: SHL-Internship Assessment
Task: Predict continuous grammar scores (1-5) from audio samples
Evaluation: Pearson Correlation & RMSE
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
from scipy.stats import pearsonr
import transformers
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer
from tqdm import tqdm
import warnings
import random
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import OneCycleLR
import gc
import re

# Suppress warnings
warnings.filterwarnings('ignore')

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ All libraries imported successfully")

# ============================================================================
# CONFIGURATION
# ============================================================================

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CONFIG = {
    # Directories
    "train_audio_dir": "/path/to/train/audios",  # UPDATE THIS
    "test_audio_dir": "/path/to/test/audios",    # UPDATE THIS
    "train_csv": "/path/to/train.csv",           # UPDATE THIS
    "test_csv": "/path/to/test.csv",             # UPDATE THIS
    "train_transcripts_csv": "/path/to/train_transcripts.csv",  # UPDATE THIS
    "test_transcripts_csv": "/path/to/test_transcripts.csv",    # UPDATE THIS
    "output_dir": "./output",
    "submission_path": "./submission.csv",
    
    # Model paths for ensemble
    "model_save_paths": [
        "./model_fold_1.pt",
        "./model_fold_2.pt",
        "./model_fold_3.pt",
        "./model_fold_4.pt",
        "./model_fold_5.pt",
    ],
    
    # Audio processing parameters
    "target_sample_rate": 16000,
    "max_audio_length": 10,
    
    # Model parameters
    "base_model": "microsoft/wavlm-base",
    "text_model": "bert-base-uncased",
    
    # Training parameters
    "batch_size": 6,
    "accumulation_steps": 2,
    "epochs": 25,
    "learning_rate": 3e-5,
    "weight_decay": 0.02,
    "use_augmentation": True,
    "augmentation_prob": 0.6,
    "use_mixed_precision": True,
    "use_swa": True,
    "swa_start_epoch": 12,
    
    # K-fold parameters
    "n_folds": 5,
    
    # Early stopping
    "early_stopping_patience": 8,
    "min_delta": 0.0001,
    
    # TTA
    "use_tta": True,
    "tta_iterations": 3,
    
    # Scheduler
    "warmup_ratio": 0.15,
    "min_lr": 1e-7,
    
    # Additional optimization flags
    "num_workers": 2,
    "pin_memory": True,
    "prefetch_factor": 2,
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
print(f"\n✓ Configuration loaded")
print(f"  - Training epochs: {CONFIG['epochs']}")
print(f"  - K-folds: {CONFIG['n_folds']}")
print(f"  - Learning rate: {CONFIG['learning_rate']}")
print(f"  - Using TTA: {CONFIG['use_tta']}")

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def advanced_audio_augmentation(waveform, sample_rate=16000, augmentation_prob=0.6):
    """Apply advanced audio augmentation techniques"""
    
    # Time stretching
    if random.random() < augmentation_prob * 0.4:
        rate = random.uniform(0.9, 1.1)
        new_length = int(waveform.shape[1] * rate)
        waveform = torch.nn.functional.interpolate(
            waveform.unsqueeze(0), 
            size=new_length, 
            mode='linear',
            align_corners=False
        ).squeeze(0)
    
    # Pitch shifting
    if random.random() < augmentation_prob * 0.3:
        shift_amount = random.uniform(-0.05, 0.05)
        waveform = waveform * (1 + shift_amount)
    
    # Add background noise
    if random.random() < augmentation_prob * 0.5:
        noise_level = random.uniform(0.001, 0.008)
        noise = torch.randn_like(waveform) * noise_level
        waveform = waveform + noise
    
    # Time shift
    if random.random() < augmentation_prob * 0.4:
        shift_samples = int(random.uniform(-0.15, 0.15) * waveform.shape[1])
        if shift_samples > 0:
            waveform = torch.cat([
                torch.zeros(1, shift_samples, device=waveform.device), 
                waveform[:, :-shift_samples]
            ], dim=1)
        elif shift_samples < 0:
            shift_samples = abs(shift_samples)
            waveform = torch.cat([
                waveform[:, shift_samples:], 
                torch.zeros(1, shift_samples, device=waveform.device)
            ], dim=1)
    
    # Volume perturbation
    if random.random() < augmentation_prob * 0.5:
        volume_factor = random.uniform(0.8, 1.2)
        waveform = waveform * volume_factor
    
    return waveform


def load_and_process_audio(file_path, target_sr=16000, max_len=10, augment=False):
    """Load and preprocess audio file with improved augmentation"""
    try:
        # Load audio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Resample if necessary
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=target_sr
            )
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Normalize
        peak = torch.abs(waveform).max()
        if peak > 0:
            waveform = waveform / peak
        
        # Apply augmentation
        if augment:
            waveform = advanced_audio_augmentation(
                waveform, target_sr, CONFIG["augmentation_prob"]
            )
            # Renormalize after augmentation
            peak = torch.abs(waveform).max()
            if peak > 0:
                waveform = waveform / peak
        
        # Fix length
        max_samples = target_sr * max_len
        if waveform.shape[1] > max_samples:
            if augment:
                start = random.randint(0, waveform.shape[1] - max_samples)
            else:
                start = (waveform.shape[1] - max_samples) // 2
            waveform = waveform[:, start:start + max_samples]
        elif waveform.shape[1] < max_samples:
            padding = torch.zeros(1, max_samples - waveform.shape[1])
            waveform = torch.cat([waveform, padding], dim=1)
        
        return waveform
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return torch.zeros(1, target_sr * max_len)

print("✓ Audio processing functions loaded")

# ============================================================================
# DATASET CLASS
# ============================================================================

class MultimodalDataset(Dataset):
    """Enhanced dataset for multimodal grammar scoring"""
    
    def __init__(self, csv_data, audio_dir, audio_feature_extractor, 
                 text_tokenizer, is_test=False, use_augmentation=False):
        
        if isinstance(csv_data, str):
            self.data = pd.read_csv(csv_data)
        else:
            self.data = csv_data.copy()
        
        self.audio_dir = audio_dir
        self.audio_feature_extractor = audio_feature_extractor
        self.text_tokenizer = text_tokenizer
        self.is_test = is_test
        self.use_augmentation = use_augmentation
        self.target_sr = CONFIG["target_sample_rate"]
        self.max_length = CONFIG["max_audio_length"]
        
        print(f"Dataset initialized: {len(self.data)} samples, "
              f"Augmentation: {use_augmentation}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Audio processing
        audio_path = os.path.join(self.audio_dir, row['filename'] + '.wav')
        waveform = load_and_process_audio(
            audio_path,
            self.target_sr,
            self.max_length,
            self.use_augmentation
        )
        
        audio_inputs = self.audio_feature_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.target_sr,
            return_tensors="pt"
        )
        
        # Text processing
        transcript = str(row.get('transcript', ''))
        text_inputs = self.text_tokenizer(
            transcript,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        result = {
            'audio_input_values': audio_inputs.input_values.squeeze(),
            'text_input_ids': text_inputs.input_ids.squeeze(),
            'text_attention_mask': text_inputs.attention_mask.squeeze(),
            'idx': idx
        }
        
        if not self.is_test:
            result['labels'] = torch.tensor(row['label_x'], dtype=torch.float)
        else:
            result['filename'] = row['filename']
        
        return result


def multimodal_collate_fn(batch):
    """Custom collate function for multimodal batching"""
    
    # Audio processing
    max_audio_len = max(x['audio_input_values'].shape[0] for x in batch)
    batch_size = len(batch)
    
    audio_input_values = torch.zeros(batch_size, max_audio_len)
    audio_attention_mask = torch.zeros(batch_size, max_audio_len)
    
    for i, item in enumerate(batch):
        audio_val = item['audio_input_values']
        length = audio_val.shape[0]
        audio_input_values[i, :length] = audio_val
        audio_attention_mask[i, :length] = 1
    
    # Text processing
    text_input_ids = torch.stack([x['text_input_ids'] for x in batch])
    text_attention_mask = torch.stack([x['text_attention_mask'] for x in batch])
    
    batch_dict = {
        'audio_input_values': audio_input_values,
        'audio_attention_mask': audio_attention_mask,
        'text_input_ids': text_input_ids,
        'text_attention_mask': text_attention_mask,
        'idx': [x['idx'] for x in batch]
    }
    
    if 'labels' in batch[0]:
        batch_dict['labels'] = torch.stack([x['labels'] for x in batch])
    else:
        batch_dict['filenames'] = [x['filename'] for x in batch]
    
    return batch_dict

print("✓ Dataset classes loaded")

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EnhancedMultimodalGrammarScoreModel(nn.Module):
    """
    Enhanced model with improvements for better RMSE:
    - Multi-head attention pooling for audio
    - Deeper projections with residual connections
    - Less aggressive freezing (only first 4 layers)
    - Enhanced cross-attention
    """
    
    def __init__(self, 
                 audio_model_name=CONFIG["base_model"],
                 text_model_name=CONFIG["text_model"]):
        super(EnhancedMultimodalGrammarScoreModel, self).__init__()
        
        # Load pretrained models
        self.audio_model = AutoModel.from_pretrained(audio_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        
        # Freeze only first 4 layers
        self._freeze_layers()
        
        audio_hidden_size = self.audio_model.config.hidden_size
        text_hidden_size = self.text_model.config.hidden_size
        
        # Multi-head attention pooling for audio
        self.audio_attention_pool = nn.MultiheadAttention(
            embed_dim=audio_hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.audio_query = nn.Parameter(torch.randn(1, 1, audio_hidden_size))
        
        # Enhanced projections
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        
        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        
        # Bidirectional cross-attention
        self.cross_attention_a2t = nn.MultiheadAttention(
            embed_dim=384, 
            num_heads=8,
            dropout=0.1, 
            batch_first=True
        )
        self.cross_attention_t2a = nn.MultiheadAttention(
            embed_dim=384, 
            num_heads=8, 
            dropout=0.1, 
            batch_first=True
        )
        
        self.cross_norm_a = nn.LayerNorm(384)
        self.cross_norm_t = nn.LayerNorm(384)
        
        # Gated fusion
        self.gate = nn.Sequential(
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Linear(384, 384),
            nn.Sigmoid()
        )
        
        # Deeper regression head
        self.regression_head = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.25),
            
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.15),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _make_key_padding_mask(self, attention_mask, target_seq_len):
        """Convert attention_mask to key_padding_mask"""
        if attention_mask is None:
            return None

        if attention_mask.ndim != 2:
            return None

        b, mask_len = attention_mask.shape
        if mask_len == target_seq_len:
            return (~attention_mask.bool())

        if mask_len % target_seq_len == 0:
            factor = mask_len // target_seq_len
            try:
                reshaped = attention_mask.view(b, target_seq_len, factor)
                chunk_has_token = reshaped.max(dim=2)[0]
                return (~chunk_has_token.bool())
            except Exception:
                return None

        return None
    
    def _freeze_layers(self):
        """Freeze only first 4 layers"""
        for name, param in self.audio_model.named_parameters():
            layer_num = self._extract_layer_num(name)
            if layer_num is not None and layer_num < 4:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        for name, param in self.text_model.named_parameters():
            if 'embeddings' in name:
                param.requires_grad = False
            elif 'encoder.layer' in name:
                match = re.search(r'encoder\.layer\.(\d+)', name)
                if match:
                    layer_num = int(match.group(1))
                    param.requires_grad = layer_num >= 4
            else:
                param.requires_grad = True
    
    def _extract_layer_num(self, name):
        match = re.search(r'layers\.(\d+)', name)
        return int(match.group(1)) if match else None
    
    def _init_weights(self):
        for module in [self.audio_projection, self.text_projection, 
                      self.regression_head, self.gate]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(self, audio_input_values, audio_attention_mask,
                text_input_ids, text_attention_mask):
        """Forward pass with enhanced attention"""

        # Process audio
        audio_outputs = self.audio_model(
            input_values=audio_input_values,
            attention_mask=audio_attention_mask
        )
        audio_hidden = audio_outputs.last_hidden_state

        key_padding_mask = self._make_key_padding_mask(
            audio_attention_mask, audio_hidden.size(1)
        )

        batch_size = audio_hidden.size(0)
        query = self.audio_query.expand(batch_size, -1, -1)

        if key_padding_mask is not None:
            audio_pooled, _ = self.audio_attention_pool(
                query, audio_hidden, audio_hidden,
                key_padding_mask=key_padding_mask
            )
        else:
            audio_pooled, _ = self.audio_attention_pool(
                query, audio_hidden, audio_hidden
            )

        audio_pooled = audio_pooled.squeeze(1)
        audio_features = self.audio_projection(audio_pooled)

        # Process text
        text_outputs = self.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        text_pooled = text_outputs.last_hidden_state[:, 0, :]
        text_features = self.text_projection(text_pooled)

        # Cross-attention
        audio_seq = audio_features.unsqueeze(1)
        text_seq = text_features.unsqueeze(1)

        audio_attended, _ = self.cross_attention_a2t(audio_seq, text_seq, text_seq)
        text_attended, _ = self.cross_attention_t2a(text_seq, audio_seq, audio_seq)

        audio_attended = self.cross_norm_a(audio_attended.squeeze(1) + audio_features)
        text_attended = self.cross_norm_t(text_attended.squeeze(1) + text_features)

        # Gated fusion
        combined = torch.cat([audio_attended, text_attended], dim=-1)
        gate_value = self.gate(combined)

        gated_audio = audio_attended * gate_value
        gated_text = text_attended * (1 - gate_value)
        final_features = torch.cat([gated_audio, gated_text], dim=-1)

        # Regression
        score = self.regression_head(final_features)
        score = torch.sigmoid(score) * 4.0 + 1.0

        return score

print("✓ Enhanced model architecture loaded")

# ============================================================================
# LOSS FUNCTIONS AND UTILITIES
# ============================================================================

class RMSEFocusedLoss(nn.Module):
    """Custom loss that focuses on RMSE optimization with ordinal awareness"""
    
    def __init__(self, mse_weight=0.5, huber_weight=0.3, ordinal_weight=0.2):
        super(RMSEFocusedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=0.5)
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        self.ordinal_weight = ordinal_weight
    
    def ordinal_loss(self, predictions, targets):
        """Encourage ordinal relationships (scores 1-5)"""
        pred_class = torch.round(predictions)
        target_class = torch.round(targets)
        class_diff = torch.abs(pred_class - target_class)
        ordinal_penalty = torch.mean(class_diff)
        return ordinal_penalty
    
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        huber_loss = self.huber(predictions, targets)
        ordinal_loss = self.ordinal_loss(predictions, targets)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.huber_weight * huber_loss +
                     self.ordinal_weight * ordinal_loss)
        
        return total_loss


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=8, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_score = -float('inf')
    
    def __call__(self, val_loss, val_score):
        score = val_score
        
        if self.best_score is None:
            self.best_score = score
            self.best_loss = val_loss
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"  Early stopping triggered!")
        else:
            self.best_score = score
            self.best_loss = val_loss
            self.counter = 0


def load_model_weights(model, checkpoint_path, device):
    """Safely load model weights handling DataParallel and SWA artifacts"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    return model

print("✓ Loss functions and utilities loaded")

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_multimodal_fold(fold, train_data, val_data, audio_feature_extractor,  
                          text_tokenizer, model_path, device):
    """Enhanced training with improved loss, scheduler, SWA, and early stopping"""
    
    print(f"\n{'='*70}")
    print(f"Training Fold {fold+1}/{CONFIG['n_folds']}")
    print(f"{'='*70}")
    
    # Create datasets
    train_dataset = MultimodalDataset(
        train_data,
        CONFIG["train_audio_dir"],
        audio_feature_extractor,
        text_tokenizer,
        use_augmentation=CONFIG.get("use_augmentation", False)
    )
    
    val_dataset = MultimodalDataset(
        val_data,
        CONFIG["train_audio_dir"],
        audio_feature_extractor,
        text_tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=multimodal_collate_fn,
        num_workers=CONFIG.get("num_workers", 4),
        pin_memory=CONFIG.get("pin_memory", True),
        prefetch_factor=CONFIG.get("prefetch_factor", 2)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        collate_fn=multimodal_collate_fn,
        num_workers=0
    )
    
    # Initialize model
    model = EnhancedMultimodalGrammarScoreModel().to(device)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # OneCycle scheduler
    steps_per_epoch = max(1, len(train_loader) // CONFIG.get("accumulation_steps", 1))
    total_steps = steps_per_epoch * CONFIG["epochs"]
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG["learning_rate"],
        total_steps=total_steps,
        pct_start=CONFIG.get("warmup_ratio", 0.1),
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )
    
    # Loss function
    criterion = RMSEFocusedLoss()
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG.get("early_stopping_patience", 8),
        min_delta=CONFIG.get("min_delta", 1e-4)
    )
    
    # SWA setup
    swa_model = None
    swa_scheduler = None
    if CONFIG.get("use_swa", False):
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(
            optimizer, 
            swa_lr=CONFIG["learning_rate"] / 10
        )
    
    # Training trackers
    best_val_corr = -1.0
    best_val_rmse = float('inf')
    train_losses = []
    val_losses = []
    val_correlations = []
    val_rmses = []
    
    scaler = GradScaler() if CONFIG.get("use_mixed_precision", False) else None
    
    # Training loop
    for epoch in range(CONFIG["epochs"]):
        # Training phase
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        progress_bar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]"
        )
        
        for i, batch in progress_bar:
            audio_input_values = batch['audio_input_values'].to(device)
            audio_attention_mask = batch['audio_attention_mask'].to(device)
            text_input_ids = batch['text_input_ids'].to(device)
            text_attention_mask = batch['text_attention_mask'].to(device)
            labels = batch['labels'].to(device).view(-1, 1)
            
            if CONFIG.get("use_mixed_precision", False):
                with autocast():
                    outputs = model(
                        audio_input_values, 
                        audio_attention_mask,
                        text_input_ids,
                        text_attention_mask
                    )
                    loss = criterion(outputs, labels) / CONFIG.get("accumulation_steps", 1)
                loss.backward()
                
                if (i + 1) % CONFIG.get("accumulation_steps", 1) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            train_loss += loss.item() * CONFIG.get("accumulation_steps", 1)
            progress_bar.set_postfix({
                'loss': f"{loss.item() * CONFIG.get('accumulation_steps', 1):.4f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # SWA update
        if CONFIG.get("use_swa", False) and epoch >= CONFIG.get("swa_start_epoch", 0):
            swa_model.update_parameters(model)
            if swa_scheduler is not None:
                swa_scheduler.step()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                audio_input_values = batch['audio_input_values'].to(device)
                audio_attention_mask = batch['audio_attention_mask'].to(device)
                text_input_ids = batch['text_input_ids'].to(device)
                text_attention_mask = batch['text_attention_mask'].to(device)
                labels = batch['labels'].to(device).view(-1, 1)
                
                outputs = model(
                    audio_input_values, 
                    audio_attention_mask,
                    text_input_ids,
                    text_attention_mask
                )
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        try:
            val_corr, _ = pearsonr(all_preds, all_labels)
        except Exception:
            val_corr = 0.0
        
        val_rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        
        val_correlations.append(val_corr)
        val_rmses.append(val_rmse)
        
        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Pearson: {val_corr:.4f}")
        print(f"  Val RMSE: {val_rmse:.4f}")
        
        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_val_rmse = val_rmse
            
            if CONFIG.get("use_swa", False) and epoch >= CONFIG.get("swa_start_epoch", 0):
                torch.save(swa_model.state_dict(), model_path)
                print(f"  ✓ Saved SWA model (Pearson: {val_corr:.4f}, RMSE: {val_rmse:.4f})")
            else:
                torch.save(model.state_dict(), model_path)
                print(f"  ✓ Saved best model (Pearson: {val_corr:.4f}, RMSE: {val_rmse:.4f})")
        
        # Early stopping check
        early_stopping(avg_val_loss, val_corr)
        if early_stopping.early_stop:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break
        
        # Memory cleanup
        torch.cuda.empty_cache()
        gc.collect()
    
    # Update BN statistics for SWA model
    if CONFIG.get("use_swa", False) and swa_model is not None:
        print("\nUpdating batch normalization statistics for SWA model...")
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        except Exception:
            torch.optim.swa_utils.update_bn(train_loader, swa_model)
        torch.save(swa_model.state_dict(), model_path)
        print("✓ SWA model finalized and saved")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Loss curves
    axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 
                    label='Train Loss', linewidth=2)
    axes[0, 0].plot(range(1, len(val_losses) + 1), val_losses, 
                    label='Val Loss', linewidth=2)
    axes[0, 0].set_title(f'Fold {fold+1} - Loss Curves', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Pearson correlation
    axes[0, 1].plot(range(1, len(val_correlations) + 1), val_correlations, 
                    linewidth=2, marker='o')
    axes[0, 1].axhline(y=best_val_corr, color='r', linestyle='--', 
                       label=f'Best: {best_val_corr:.4f}')
    axes[0, 1].set_title(f'Fold {fold+1} - Validation Pearson Correlation', 
                         fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Pearson r')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMSE curve
    axes[1, 0].plot(range(1, len(val_rmses) + 1), val_rmses, 
                    linewidth=2, marker='s')
    axes[1, 0].axhline(y=best_val_rmse, color='r', linestyle='--', 
                       label=f'Best: {best_val_rmse:.4f}')
    axes[1, 0].set_title(f'Fold {fold+1} - Validation RMSE', 
                         fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('RMSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1, 1].scatter(all_labels, all_preds, alpha=0.6, s=40)
    axes[1, 1].plot([1, 5], [1, 5], 'r--', label='Perfect Prediction', linewidth=2)
    axes[1, 1].set_xlabel('Actual Grammar Score', fontsize=11)
    axes[1, 1].set_ylabel('Predicted Grammar Score', fontsize=11)
    axes[1, 1].set_title(
        f'Fold {fold+1} - Final Predictions\n(r={val_corr:.4f}, RMSE={val_rmse:.4f})', 
        fontsize=12, fontweight='bold'
    )
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0.5, 5.5])
    axes[1, 1].set_ylim([0.5, 5.5])
    
    plt.tight_layout()
    plt.savefig(
        os.path.join(CONFIG["output_dir"], f'multimodal_fold_{fold+1}.png'), 
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    
    print(f"✓ Fold {fold+1} complete - Best Pearson: {best_val_corr:.4f}, "
          f"Best RMSE: {best_val_rmse:.4f}\n")
    
    return best_val_corr, best_val_rmse, all_labels, all_preds

print("✓ Training function loaded")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("=" * 70)
    print("ENHANCED MULTIMODAL GRAMMAR SCORE PREDICTION")
    print("=" * 70)

    # Load data
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv(CONFIG["train_csv"])
    
    # Load transcripts
    print("[2/5] Loading transcripts...")
    train_transcripts = pd.read_csv(CONFIG["train_transcripts_csv"])
    
    # Merge transcripts
    train_df = train_df.merge(train_transcripts, on="filename", how="left")
    train_df["transcript"] = train_df["transcript"].fillna("")
    
    print(f"\nTraining data shape: {train_df.shape}")
    print(f"\nLabel distribution:")
    print(train_df["label_x"].describe())
    
    # Initialize feature extractors
    print("\n[3/5] Initializing feature extractors...")
    audio_feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG["base_model"])
    text_tokenizer = AutoTokenizer.from_pretrained(CONFIG["text_model"])
    print("✓ Feature extractors loaded")
    
    # K-fold cross-validation training
    print(f"\n[4/5] Starting {CONFIG['n_folds']}-fold cross-validation...")
    print("=" * 70)
    kf = KFold(n_splits=CONFIG["n_folds"], shuffle=True, random_state=SEED)
    
    fold_correlations = []
    fold_rmses = []
    all_val_labels = []
    all_val_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        fold_train = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val = train_df.iloc[val_idx].reset_index(drop=True)
        
        best_corr, best_rmse, val_labels, val_preds = train_multimodal_fold(
            fold,
            fold_train,
            fold_val,
            audio_feature_extractor,
            text_tokenizer,
            CONFIG["model_save_paths"][fold],
            device
        )
        
        fold_correlations.append(best_corr)
        fold_rmses.append(best_rmse)
        
        all_val_labels.extend(np.array(val_labels).flatten().tolist())
        all_val_preds.extend(np.array(val_preds).flatten().tolist())
    
    # Compute overall CV metrics
    all_val_labels = np.array(all_val_labels)
    all_val_preds = np.array(all_val_preds)
    
    if len(all_val_labels) > 1:
        overall_corr, _ = pearsonr(all_val_labels, all_val_preds)
    else:
        overall_corr = 0.0
    
    overall_rmse = np.sqrt(mean_squared_error(all_val_labels, all_val_preds))
    
    # Print CV summary
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 70)
    for i in range(CONFIG["n_folds"]):
        print(f"Fold {i+1}: Pearson = {fold_correlations[i]:.4f}, "
              f"RMSE = {fold_rmses[i]:.4f}")
    print("-" * 70)
    print(f"Mean Pearson: {np.mean(fold_correlations):.4f} "
          f"(±{np.std(fold_correlations):.4f})")
    print(f"Mean RMSE: {np.mean(fold_rmses):.4f} "
          f"(±{np.std(fold_rmses):.4f})")
    print("-" * 70)
    print(f"Overall CV Pearson: {overall_corr:.4f}")
    print(f"Overall CV RMSE: {overall_rmse:.4f}")
    print("=" * 70)
    
    # Save CV summary plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(1, CONFIG["n_folds"] + 1), fold_correlations,
                alpha=0.8, edgecolor='black')
    axes[0].axhline(np.mean(fold_correlations), color='r', linestyle='--',
                    label=f"Mean: {np.mean(fold_correlations):.4f}")
    axes[0].set_title("Pearson by Fold")
    axes[0].set_xlabel("Fold")
    axes[0].set_ylabel("Pearson Correlation")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(1, CONFIG["n_folds"] + 1), fold_rmses,
                alpha=0.8, edgecolor='black')
    axes[1].axhline(np.mean(fold_rmses), color='r', linestyle='--',
                    label=f"Mean: {np.mean(fold_rmses):.4f}")
    axes[1].set_title("RMSE by Fold")
    axes[1].set_xlabel("Fold")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "cv_summary.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n[5/5] Training complete!")
    print(f"✓ All outputs saved to: {CONFIG['output_dir']}")
    print(f"\n{'='*70}")
    print("KEY METRICS FOR SUBMISSION")
    print("=" * 70)
    print(f"Training CV Pearson: {overall_corr:.4f}")
    print(f"Training CV RMSE: {overall_rmse:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main(), labels) / CONFIG.get("accumulation_steps", 1)
                
                scaler.scale(loss).backward()
                
                if (i + 1) % CONFIG.get("accumulation_steps", 1) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(
                    audio_input_values, 
                    audio_attention_mask,
                    text_input_ids,
                    text_attention_mask
                )
                loss = criterion(outputs