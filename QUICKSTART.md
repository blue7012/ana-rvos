# QUICKSTART.md — Getting Started

## What This Project Implements

**Paper**: "Actor and Action Video Segmentation from a Sentence" (Gavrilyuk et al., CVPR 2018)

**Task**: Given a video and a text description (e.g., "a person running"), output a pixel-level mask showing the actor and action in each frame.

**Key Innovation**: Uses dynamic convolutional filters generated from text embeddings to adapt the visual processing to the specific query.

---

## Project Structure

\\\
ana-rvos/
├── README.md              # Full project overview & specifications
├── ARCHITECTURE.md        # Detailed model architecture & data flow
├── QUICKSTART.md          # This file
├── .claude/CLAUDE.md      # ML engineering standards (READ FIRST)
├── Gavrilyuk_Actor_and_Action_CVPR_2018_paper.pdf   # Main paper
├── 1603.06180v1.pdf       # Reference work (Hu et al., 2016)
└── [To be created]
    ├── config.py          # Dataclass configurations
    ├── dataset.py         # Data loading (video frames, text, masks)
    ├── preprocessing.py   # Video/flow utilities
    ├── model/
    │   ├── layers.py      # Atomic components (encoders, decoders)
    │   ├── modules.py     # Functional modules
    │   └── model.py       # Full ActorActionVideoSegmentation
    ├── training/
    │   ├── loss.py        # Loss functions
    │   ├── optimizer.py   # Adam setup
    │   └── trainer.py     # Training loop
    ├── evaluation/
    │   ├── metrics.py     # IoU, F-measure, etc.
    │   └── inference.py   # Single-example inference
    ├── tests/             # Unit tests
    ├── train.py           # Entry point: python train.py
    ├── evaluate.py        # Entry point: python evaluate.py
    └── infer.py           # Entry point: python infer.py
\\\

---

## Implementation Phases

### Phase 1: Configuration (Ready to start)
- Create \config.py\ with dataclass configurations
- Define all model hyperparameters (no magic numbers)

### Phase 2: Data Pipeline
- Load video frames + pre-computed optical flow
- Tokenize sentences → Word2Vec embeddings (fixed)
- Load segmentation masks

### Phase 3: Model Components
- TextualEncoder: 1D Conv on Word2Vec
- VideoEncoder: I3D (two streams: RGB, Flow)
- Decoder: Dynamic filters + upsampling

### Phase 4: Full Model Assembly
- Combine encoders → decoder
- Two-stream fusion (averaging)

### Phase 5: Training
- Binary cross-entropy loss
- Optimizer: Adam
- Validation loop with checkpoint saving

### Phase 6: Evaluation & Inference
- Compute IoU metrics
- Single-video inference script

---

## Critical Rules (From .claude/CLAUDE.md)

1. **Tensor Shapes**: Every function MUST document input/output shapes in docstrings
2. **No Magic Numbers**: All dimensions in \config.py\, referenced by name
3. **Complexity Estimates**: Note time/space for major functions
4. **Use einops**: For tensor reshaping (clearer than view/permute)
5. **Paper Fidelity**: Every class/function cites its paper section (§3.1, etc.)

Example:
\\\python
def dynamic_convolution(text_emb: Tensor, feature_map: Tensor) -> Tensor:
    '''
    Gavrilyuk et al., §3.3 — Decoder with Dynamic Filters
    
    Args:
        text_emb: (B, 300) sentence embedding
        feature_map: (B, H, W, C) spatial features
    
    Returns:
        response: (B, H, W, C_out) segmentation response
    
    Complexity: O(B * H * W * C * C_out) time | O(B * H * W * C_out) memory
    '''
\\\

---

## Key Implementation Details

### 1. Word2Vec Embeddings
- **Pre-trained**: Google News (300-dim)
- **Frozen**: NO fine-tuning during training
- **Path**: Load at startup, reuse for all batches

### 2. I3D Pre-training
- **Two models**: Separate RGB and Flow I3D networks
- **Pre-trained on**: ImageNet + Kinetics dataset
- **Output**: Inception block (832-dim features)

### 3. Optical Flow
- **Pre-computed**: Assume flows already computed (TV-L1, FlowNet2, etc.)
- **Format**: 2-channel (u, v) velocity fields
- **Normalization**: Clip to [-20, 20] range

### 4. Dynamic Filters (Core Innovation)
- **Generated from**: Text embedding via FC layer
- **Applied at**: 3 decoder levels (128, 256, 832 filter sizes)
- **Activation**: tanh + L2-normalization

### 5. Two-Stream Fusion
- **RGB stream**: I3D on RGB frames
- **Flow stream**: I3D on optical flow
- **Fusion**: Element-wise averaging of response maps

---

## Ambiguities from Paper (Assumptions Made)

| Issue | Assumption |
|-------|-----------|
| Loss function | Binary Cross-Entropy per pixel |
| Decoder upsampling | Transpose convolution (deconvolution) |
| 1D Conv output dim | Same as input (300-dim) or larger |
| I3D fine-tuning | Frozen or low-LR fine-tune (standard) |
| Flow normalization | [-20, 20] (standard range) |
| Batch norm placement | After conv, before activation |

---

## Expected Results

- **Segmentation IoU**: Typically 60-70% on video segmentation (from paper)
- **Generalization**: Model should segment unseen actor/action pairs via pre-trained embeddings

---

## Dependencies

\\\
torch >= 2.0
torchvision
numpy
opencv-python (video I/O)
einops (tensor operations)
wandb or tensorboard (optional, for tracking)
\\\

**Pre-trained Models to Download**:
- I3D (RGB and Flow variants)
- Word2Vec (Google News)

---

## Before You Start

1. **Read .claude/CLAUDE.md** — Establishes ML engineering standards
2. **Read README.md** — Full project specifications
3. **Read ARCHITECTURE.md** — Detailed model architecture & data flow
4. **Check the PDFs** — Original papers in the repo

---

## Questions During Implementation

### 🔴 CRITICAL (Stop & ask if unsure)
- Loss function design
- Dynamic filter application (broadcast semantics)
- I3D output handling (spatial dims, channels)

### 🟡 IMPORTANT (Use sensible default if unsure)
- Batch normalization placement
- Learning rate schedule
- Data augmentation strategy

### 🟢 MINOR (Pick a default, move on)
- Weight initialization
- Exact upsampling stride factors
- Logging frequency

---

## Running the Project (Once Implemented)

\\\ash
# Training
python train.py --config config.yaml --device cuda:0 --seed 42

# Evaluation
python evaluate.py --checkpoint best_model.pt --dataset a2d_test

# Inference on single video
python infer.py --video video.mp4 --sentence "a person running" --checkpoint best_model.pt --output mask.png
\\\

---

## Validation Checklist

Before committing code:
- [ ] Single batch overfits (loss → 0 in 100 steps)
- [ ] Validation metrics improve over epochs
- [ ] Output shapes match expected (B, T, H, W, 1)
- [ ] All hyperparameters in \config.py\
- [ ] Every function has docstring with shapes
- [ ] No magic numbers in code

---

**Status**: Ready for Phase 1  
**Next**: Create \config.py\ with ModelConfig, TrainingConfig, DatasetConfig  
**Questions?**: Check README.md, ARCHITECTURE.md, or the papers
