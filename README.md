# Actor and Action Video Segmentation from a Sentence

## Project Overview

This project implements the paper:
- **Title**: "Actor and Action Video Segmentation from a Sentence"
- **Authors**: Kirill Gavrilyuk, Amir Ghodrati, Zhenyang Li, Cees G. M. Snoek
- **Conference**: CVPR 2018
- **ArXiv**: 1803.07485
- **Reference Work**: "Segmentation from Natural Language Expressions" (Hu et al., 2016, arXiv 1603.06180)

### What the Project Does

The model performs **pixel-level spatio-temporal segmentation** of actors and their actions in video based on natural language descriptions.

**Key Innovation**: Unlike prior work limited to fixed vocabularies (e.g., predefined "person", "walking"), this method:
- Accepts open-ended natural language queries (e.g., "a man in dark suit standing on the back")
- Segments the specified actor and action in every frame of the video
- Generalizes to actor/action pairs outside the training vocabulary via pre-trained word embeddings

**Example Application**: Given a video and the sentence "a person running", the model outputs a pixel-level mask of the running person in each frame.

---

## Model Architecture

### Three Main Components

1. **Textual Encoder** (Section 3.1)
   - Encodes natural language sentence into fixed-size vector
   - Method: Pre-trained Word2Vec embeddings (Google News, 300-dim, frozen)
   - Network: 1D Convolutional layer (kernel=2) + ReLU + max-pooling
   - Output: Fixed-size sentence representation

2. **Video Encoder** (Section 3.2)
   - Extracts spatio-temporal features from video
   - Method: I3D (Inflated 3D Convolutions, Carreira & Zisserman 2017)
   - Pre-training: ImageNet (image classification) + Kinetics (action recognition)
   - Two streams: RGB video + Optical Flow (motion)
   - Processing:
     - I3D forward pass → Inception block output (before final max-pool)
     - Temporal average pooling → spatial feature map (H, W, 832)
     - L2-normalize per spatial location
     - Append spatial coordinates (x, y) → enable spatial reasoning ("left", "above")
   - Output: (H, W, 834) spatial feature map with spatial context

3. **Decoder with Dynamic Filters** (Section 3.3)
   - Generates segmentation mask from text embedding + visual features
   - Novel Component: **Dynamic filters** generated from sentence embedding
   - Process:
     - FC layer converts text embedding → dynamic convolutional filters
     - Filters applied at multiple spatial resolutions (3 levels: 32×32, 128×128, 512×512)
     - Each level: dynamic conv + deconvolution upsampling
     - Activation: tanh + L2-normalization on features
   - Output: (H_original, W_original, 1) binary segmentation mask per frame

### Two-Stream Fusion
- RGB and Flow streams processed independently with separate I3D models
- Fusion: Element-wise averaging of response maps (or concatenation)
- Rationale: Appearance + motion complement each other for action understanding

### Key Architectural Difference from Image-Based Predecessor (Hu et al.)

| Aspect | Hu et al. (Static Image) | Gavrilyuk et al. (Video) |
|--------|---|---|
| Input | Single image + sentence | Video sequence + sentence |
| Vision Backbone | 2D FCN (VGGNet) | I3D (3D Convolutional) |
| Text Encoding | LSTM network | 1D Convolution |
| Dynamic Filters | Not used | Core component (per resolution) |
| Motion Stream | N/A | Yes (optical flow) |
| Temporal Handling | N/A | Average pooling over time |

---

## Training & Evaluation

### Datasets
- **A2D (Actor-Action Dataset)**: 43 predefined actor-action pairs, extended with 7,500+ natural language descriptions
- **J-HMDB**: Human action dataset, similarly extended with sentence annotations
- **Split**: Train/val/test (paper specifies ratios, to be confirmed)

### Training Setup
- **Objective**: Supervised learning (pixel-level binary segmentation)
- **Loss**: Binary Cross-Entropy per pixel (paper doesn't explicitly specify; standard choice)
- **Optimizer**: Adam (standard, paper doesn't override)
- **End-to-End**: All components trained jointly via backpropagation

### Evaluation Metrics
- **Intersection over Union (IoU)**: Per-frame, then averaged over video
- **Mean IoU**: Across all test videos
- **Boundary F-measure**: (Optional) contour accuracy

### Expected Results
- Paper reports significant improvements over baselines
- Typical segmentation IoU on video: 60-70% (ballpark, to be confirmed from paper tables)

---

## Implementation Plan

### Phase 1: Configuration & Utilities
**Files to create**: 
- config.py — ModelConfig, TrainingConfig, DatasetConfig (dataclasses)
- constants.py — hardcoded paths, model dimensions

**Content**:
`python
@dataclass
class ModelConfig:
    # Textual Encoder
    word_embedding_dim: int = 300           # Pre-trained Word2Vec
    text_encoder_output_dim: int = 300      # After 1D conv (paper doesn't specify, assume output of conv)
    
    # Video Encoder (I3D)
    i3d_feature_dim: int = 832              # Inception block output
    spatial_dim: int = 28                   # Spatial resolution after I3D (e.g., 512/16 for stride=16)
    
    # Decoder
    dynamic_filter_sizes: list = [128, 256, 832]  # Per resolution level
    output_height: int = 512
    output_width: int = 512
    
    # Spatial coordinates (appended to features)
    use_spatial_coords: bool = True
    
@dataclass
class TrainingConfig:
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 50
    optimizer: str = "adam"
`

### Phase 2: Data Loading
**Files**: dataset.py, preprocessing.py

**Tasks**:
- Video frame loading (RGB)
- Optical flow loading (pre-computed, e.g., TV-L1 or FlowNet2)
- Sentence tokenization → Word2Vec embedding lookup (fixed weights, no fine-tuning)
- Segmentation mask loading (binary ground truth per frame)
- PyTorch DataLoader with temporal batching (frames × batch_size)

**Key constraint**: Word2Vec embeddings must be frozen (not fine-tuned)

### Phase 3: Core Modules
**Files**: model/layers.py, model/modules.py

Implement atomic building blocks:
- TextualEncoder — 1D CNN on Word2Vec embeddings
- VideoEncoder — I3D feature extractor (separate for RGB and Flow)
- DynamicConvolutionLayer — Generate + apply dynamic filters
- DecoderBlock — Multi-resolution decoder with upsampling

### Phase 4: Full Model
**File**: model/model.py

`python
class ActorActionVideoSegmentation(nn.Module):
    """
    Paper: Gavrilyuk et al., CVPR 2018, Section 3
    
    Inputs:
        video_rgb: (B, T, H, W, 3) RGB frames
        video_flow: (B, T, H, W, 2) Optical flow
        sentences: (B, max_words) word indices
    
    Output:
        segmentation: (B, T, H, W, 1) binary mask ∈ [0, 1]
    
    Architecture:
        1. Text encoder: sentence → embedding
        2. Video encoder (two streams): RGB + Flow → spatial features
        3. Decoder: text embedding + features → segmentation
    """
`

### Phase 5: Training Loop
**Files**: 	raining/loss.py, 	raining/trainer.py

- Loss: Binary cross-entropy (or alternatives: Dice loss, IoU loss)
- Optimizer: Adam with learning rate scheduling (optional)
- Validation: Compute IoU every N epochs
- Checkpoint: Save best model by validation IoU

### Phase 6: Evaluation & Inference
**Files**: evaluation/metrics.py, inference.py

- Metric computation: IoU, F-measure
- Inference script: Load model, run on single video+sentence, output mask
- Visualization: Display segmented frames

### Phase 7: Testing & Validation
**Files**: 	ests/test_shapes.py, 	ests/test_overfit.py

- Unit tests: Tensor shapes through each module
- Single-batch overfit test: Model should fit one example (loss → 0)
- Sanity checks: Edge cases (zero embeddings, constant features)

---

## Critical Implementation Notes

### 1. Word2Vec Embedding (FIXED)
- Source: Google News pre-trained, 300-dimensional
- **MUST NOT fine-tune** during training
- Handling OOV words: map to zero vector (or UNK token)
- Load once at startup, reuse for all batches

### 2. I3D Pre-trained Weights
- Two separate models: RGB and Flow
- Pre-trained on ImageNet + Kinetics dataset
- Take output from Inception block (before final global avg-pool)
- Standard: freeze initial layers, fine-tune later layers (or freeze entirely if data is limited)

### 3. Optical Flow Computation
- Paper assumes pre-computed flows (e.g., TV-L1, FlowNet2, PWCNet)
- Not part of this implementation (assume flows provided)
- Format: 2-channel (u, v) velocity fields
- Normalization: Clip to [-20, 20] range typical for optical flow

### 4. Dynamic Filters (Core Innovation)
- Text embedding (e.g., 300-dim) → FC layer → filters for each resolution
- Sizes: 128, 256, 832 dimensions
- Applied as 1×1 convolutions with dynamic kernels
- Activation: tanh (paper specifies)
- L2-normalize output features

### 5. Spatial Coordinates
- Each spatial location (i, j) gets appended (x, y) coordinates
- Normalization: x, y ∈ [0, 1] (normalized by image dimensions)
- Purpose: Enable spatial reasoning ("left of", "above", etc.)
- Implementation: Create coordinate grids, normalize, concatenate to features

### 6. Data Augmentation
- Conservative approach (limited in paper):
  - Horizontal flip (both RGB and flow consistent)
  - Avoid temporal augmentation (breaks flow temporal coherence)
- Normalization: ImageNet stats for RGB; zero-mean for flow

---

## Ambiguities from Paper (To Clarify During Implementation)

| Level | Issue | Current Assumption | Resolution |
|---|---|---|---|
| 🔴 **CRITICAL** | Loss function not specified in paper | Binary cross-entropy per pixel | Confirm from Eq. / use standard |
| 🔴 **CRITICAL** | Decoder upsampling method | Transpose convolution (deconvolution) | Paper shows in Fig. 2 |
| 🟡 **IMPORTANT** | 1D Conv output dimension | Likely same as input (300-dim) or larger | Infer from ablation if provided |
| 🟡 **IMPORTANT** | I3D fine-tuning strategy | Assume frozen or low-LR fine-tune | Check paper's experimental setup |
| 🟡 **IMPORTANT** | Flow normalization range | [-20, 20] (standard optical flow) | Confirm with dataset stats |
| 🟢 **MINOR** | Batch normalization placement | PyTorch defaults (after conv, before activation) | Not critical for results |
| 🟢 **MINOR** | Weight initialization | Kaiming uniform (conv), normal (FC) | Standard PyTorch defaults |

---

## Reproducibility & Best Practices

### Random Seed Management
`python
import torch, numpy as np, random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
`

### Configuration & Logging
- All hyperparameters in config.py (no magic numbers)
- Log config, train/val metrics, checkpoints
- Track experiments with wandb or tensorboard

### Validation Protocol
- Train/val/test split fixed per paper
- Validation every N epochs, save best model by mIoU
- Test set used only for final evaluation (single run, no tuning)

---

## File Structure (Target)

`
ana-rvos/
├── README.md                    # This file
├── CLAUDE.md                    # (Read-only ML engineering standards)
├── config.py                    # Dataclass configs
├── constants.py                 # Hardcoded constants
├── dataset.py                   # DataLoader implementation
├── preprocessing.py             # Video/flow loading utilities
├── model/
│   ├── __init__.py
│   ├── layers.py               # Atomic components
│   ├── modules.py              # Encoder/Decoder modules
│   └── model.py                # Full model
├── training/
│   ├── __init__.py
│   ├── loss.py                 # Loss functions
│   ├── trainer.py              # Training loop
│   └── optimizer.py            # Optimizer setup
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py              # IoU, F-measure, etc.
│   └── inference.py            # Single-sample inference
├── tests/
│   ├── test_shapes.py          # Unit tests
│   ├── test_overfit.py         # Single-batch test
│   └── test_sanity.py          # Edge case tests
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
└── infer.py                     # Inference script
`

---

## Running the Project

### Expected Commands (to be implemented)

`ash
# Training
python train.py --config configs/config.yaml --device cuda:0 --seed 42

# Evaluation
python evaluate.py --checkpoint checkpoints/best_model.pt --dataset a2d_test

# Inference
python infer.py --video data/sample.mp4 --sentence "a man running" --checkpoint checkpoints/best_model.pt --output results/mask.png
`

---

## References

1. **Gavrilyuk, K., Ghodrati, A., Li, Z., & Snoek, C. G. M.** (2018).  
   *Actor and Action Video Segmentation from a Sentence*. CVPR 2018.  
   https://arxiv.org/abs/1803.07485

2. **Hu, R., Rohrbach, M., & Darrell, T.** (2016).  
   *Segmentation from Natural Language Expressions*. ECCV 2016.  
   https://arxiv.org/abs/1603.06180

3. **Carreira, J., & Zisserman, A.** (2017).  
   *Quo Vadis, Action Recognition? A New Model and Large-Scale Datasets*. CVPR 2017.  
   https://arxiv.org/abs/1705.07971

4. **Xu, B., Fu, Y., Jiang, Y. G., Li, B., & Sigal, L.** (2015).  
   *Actor and Action Video Segmentation from a Sentence*. CVPR 2015.  
   (A2D Dataset Paper) https://arxiv.org/abs/1411.4928

---

**Status**: Phase 1 Ready (Config & Structure)  
**Created**: June 7, 2026  
**Target**: Complete implementation following CLAUDE.md ML standards
