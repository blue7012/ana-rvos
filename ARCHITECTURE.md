# ARCHITECTURE.md — Actor and Action Video Segmentation Model

## System Architecture Diagram

\\\
Input: Video + Sentence
    |
    +----> Textual Encoder (§3.1)
    |       - Word2Vec lookup (300-dim, frozen)
    |       - 1D Conv (kernel=2) + ReLU + max-pool
    |       → sentence_embedding (B, 300)
    |
    +----> Video Encoder RGB (§3.2)
    |       - I3D backbone (pre-trained ImageNet + Kinetics)
    |       - Input: (B, T, H, W, 3) RGB frames
    |       - Output: (B, H', W', 832) spatial features
    |       - L2-normalize + append spatial coords (x, y)
    |       → feat_rgb (B, H', W', 834)
    |
    +----> Video Encoder Flow (§3.2)
    |       - I3D backbone (pre-trained on Flow data)
    |       - Input: (B, T, H, W, 2) Optical flow
    |       - Output: (B, H', W', 832) spatial features
    |       → feat_flow (B, H', W', 834)
    |
    +----> Stream Fusion
    |       - Element-wise average: feat = (feat_rgb + feat_flow) / 2
    |       → feat_fused (B, H', W', 834)
    |
    +----> Decoder with Dynamic Filters (§3.3)
    |       Level 1: 32×32 resolution
    |       - Dynamic filter from text: text_emb → (B, 1, 1, 128)
    |       - Conv2D: feat(B,32,32,834) ⊗ filter → (B, 32, 32, 128)
    |       - Activation: tanh + L2-norm
    |       - Deconv upsample → (B, 128, 128, 256)
    |
    |       Level 2: 128×128 resolution
    |       - Dynamic filter: text_emb → (B, 1, 1, 256)
    |       - Conv2D: feat_up ⊗ filter → (B, 128, 128, 256)
    |       - Deconv upsample → (B, 512, 512, 128)
    |
    |       Level 3: 512×512 resolution (final)
    |       - Dynamic filter: text_emb → (B, 1, 1, 832)
    |       - Conv2D: feat_up ⊗ filter → (B, 512, 512, 832)
    |       - Final 1×1 conv → segmentation logits (B, 512, 512, 1)
    |
    └----> Output: Segmentation Mask
            - Shape: (B, T, H, W, 1)
            - Values: [0, 1] (after sigmoid)
            - Interpretation: Probability of actor+action at each pixel
\\\

---

## Module Dependency Graph

\\\
config.py
    ↓
dataset.py → preprocessing.py
    ↓
model/
    ├── layers.py (atomic modules)
    │   ├── TextualEncoder
    │   ├── I3DFeatureExtractor
    │   ├── DynamicConvBlock
    │   └── DecoderBlock
    │
    ├── modules.py (functional components)
    │   ├── SpatialCoordinates
    │   ├── L2Normalize
    │   └── DynamicConvLayer
    │
    └── model.py (full architecture)
        └── ActorActionVideoSegmentation
                ├── uses TextualEncoder
                ├── uses VideoEncoder(I3D) ×2 (RGB, Flow)
                └── uses Decoder

training/
    ├── loss.py (binary segmentation loss)
    ├── optimizer.py (Adam, scheduler)
    └── trainer.py (training loop)
        └── imports model.py, loss.py, dataset.py

evaluation/
    ├── metrics.py (IoU, F-measure)
    └── inference.py (single example inference)

tests/
    ├── test_shapes.py
    ├── test_overfit.py
    └── test_sanity.py

train.py, evaluate.py, infer.py (entry points)
\\\

---

## Data Flow During Training

\\\
DataLoader
    ↓
Batch: {'video_rgb': (B, T, H, W, 3),
        'video_flow': (B, T, H, W, 2),
        'sentences': (B, max_words),
        'masks_gt': (B, T, H, W, 1)}
    ↓
Model Forward Pass
    ├── text_emb = TextualEncoder(sentences)     → (B, 300)
    ├── feat_rgb = VideoEncoder_RGB(video_rgb)   → (B, H', W', 834)
    ├── feat_flow = VideoEncoder_Flow(video_flow)→ (B, H', W', 834)
    ├── feat = (feat_rgb + feat_flow) / 2        → (B, H', W', 834)
    └── pred_masks = Decoder(text_emb, feat)    → (B, T, H, W, 1)
    ↓
Loss Computation
    loss = BCEWithLogitsLoss(pred_masks, masks_gt)
    ↓
Backward Pass
    loss.backward()
    ↓
Optimizer Step
    optimizer.step()
    ↓
Validation (every N epochs)
    ↓
Checkpoint Save (if validation mIoU improves)
\\\

---

## Tensor Shapes Through the Model

### Textual Encoder (§3.1)

| Stage | Operation | Shape |
|-------|-----------|-------|
| Input | Raw sentences | (B, max_words) |
| Word2Vec lookup | Fixed embeddings | (B, max_words, 300) |
| 1D Conv | kernel=2, 300 channels | (B, max_words-1, 300) |
| ReLU | Activation | (B, max_words-1, 300) |
| Max Pool | Over time axis | (B, 300) |
| **Output** | **Sentence embedding** | **(B, 300)** |

### Video Encoder (§3.2)

| Stage | Operation | Shape (RGB) |
|-------|-----------|-------|
| Input | Raw video frames | (B, T, H, W, 3) |
| I3D Conv | Stem layers | (B, T', H/4, W/4, 64) |
| I3D Conv | Inception blocks | (B, T', H/8, W/8, 256) |
| I3D Conv | More blocks | (B, T', H/16, W/16, 832) |
| Temporal Avg Pool | Over time | (B, H/16, W/16, 832) |
| L2 Normalize | Per spatial location | (B, H/16, W/16, 832) |
| Append Coords | (x, y) channels | (B, H/16, W/16, 834) |
| **Output** | **Spatial feature map** | **(B, 28, 28, 834)** |

*Assuming H=W=512, I3D stride=16*

### Decoder with Dynamic Filters (§3.3)

| Level | Text→Filter | Input Features | Dynamic Conv | Deconv | Output |
|-------|---|---|---|---|---|
| 1 | FC(300→128) | (B, 28, 28, 834) | (B, 28, 28, 128) | 32×32 stride=2 | (B, 56, 56, 128) |
| 2 | FC(300→256) | (B, 56, 56, ?) | (B, 56, 56, 256) | 128×128 stride=2 | (B, 112, 112, 256) |
| 3 | FC(300→832) | (B, 112, 112, ?) | (B, 112, 112, 832) | 512×512 stride~4 | (B, 512, 512, 1) |
| **Output** | | | | | **(B, 512, 512, 1)** |

---

## Key Computational Patterns

### 1. Dynamic Convolution
\\\python
# Simplified pseudocode
text_embedding: (B, 300)
feature_map: (B, H, W, C)

# Generate dynamic kernel
dynamic_kernel = fc_layer(text_embedding)  # (B, C_out)
dynamic_kernel = dynamic_kernel.view(B, 1, 1, C_out)  # (B, 1, 1, C_out)

# Apply as 1×1 convolution (broadcast across spatial dims)
response = F.conv2d(feature_map, dynamic_kernel)  # (B, H, W, C_out)
response = torch.tanh(response)
response = F.normalize(response, p=2, dim=-1)
\\\

### 2. Spatial Coordinate Appending
\\\python
B, H, W, C = feature_map.shape

# Create normalized coordinate grids
x = torch.linspace(0, 1, W).to(device)
y = torch.linspace(0, 1, H).to(device)
xx, yy = torch.meshgrid(x, y, indexing='xy')
xx = xx.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)
yy = yy.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1, 1)

# Append to features
feature_with_coords = torch.cat([feature_map, xx, yy], dim=-1)  # (B, H, W, C+2)
\\\

### 3. Two-Stream Fusion
\\\python
feat_rgb = video_encoder_rgb(video_rgb)    # (B, H, W, 834)
feat_flow = video_encoder_flow(video_flow)  # (B, H, W, 834)

# Fusion strategies (paper uses averaging)
feat_fused = (feat_rgb + feat_flow) / 2    # Element-wise average
# Alternative: torch.cat([feat_rgb, feat_flow], dim=-1)  # Concatenation
\\\

---

## Memory & Computation Estimates

### Per Video (T=8 frames, H=W=512)

| Component | FLOPs | Memory |
|-----------|-------|--------|
| TextualEncoder (1D Conv, 20 words) | ~10 KFLOP | ~1 MB |
| VideoEncoder RGB (I3D, T=8) | ~100 GFLOP | ~2 GB (inference), ~8 GB (training with gradients) |
| VideoEncoder Flow (I3D, T=8) | ~100 GFLOP | ~2 GB (inference), ~8 GB (training) |
| Decoder (3 levels, dynamic conv) | ~50 GFLOP | ~500 MB |
| **Total** | **~250 GFLOP** | **~16 GB (training batch size 1)** |

**Batch Size 4 would require ~64 GB GPU memory** → Consider:
- Gradient checkpointing
- Mixed precision (FP16)
- Reduce video resolution or frames per batch

---

## Activation Functions Used

| Layer | Activation | Reason |
|-------|-----------|--------|
| TextualEncoder 1D Conv | ReLU | Standard for text encoding |
| I3D (from paper) | ReLU (internal) | Standard for vision |
| Dynamic Filter Output | tanh | Paper specifies (§3.3), provides [-1, 1] range |
| Final Segmentation | Sigmoid (post-processing) | Binary classification [0, 1] |

---

## Normalization Strategies

| Layer | Normalization | Note |
|-------|---|---|
| Word2Vec embeddings | None (frozen) | Pre-trained, not updated |
| I3D features (spatial) | L2 per-location (§3.2) | Self-normalization for spatial coords |
| Dynamic filter output | L2 normalize (§3.3) | Paper specifies |
| Final mask logits | None (raw) | Sigmoid applied during loss (BCE with logits) |

---

## Comparison: Hu et al. (Image) vs. Gavrilyuk et al. (Video)

\\\
┌─────────────────────────────────────────────────────────────┐
│ Hu et al. (2016) — Image Segmentation from Sentence      │
├─────────────────────────────────────────────────────────────┤
│ Input:    Image + sentence                                  │
│ Vision:   2D FCN (VGGNet backbone, no temporal)             │
│ Text:     LSTM encoder                                      │
│ Decoder:  Upsampling via deconvolution (fixed filters)      │
│ Output:   Single segmentation mask                          │
└─────────────────────────────────────────────────────────────┘
                           ↑
                        [Extended to]
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Gavrilyuk et al. (2018) — Video Segmentation from Sentence │
├─────────────────────────────────────────────────────────────┤
│ Input:    Video (RGB + Flow) + sentence                     │
│ Vision:   I3D (3D conv, spatio-temporal)                    │
│ Text:     1D CNN encoder (simpler, more efficient)          │
│ Dynamic:  Filters generated from text (novel!)              │
│ Decoder:  Multiple resolution levels + dynamic conv         │
│ Output:   Per-frame segmentation masks (T frames)           │
│ Fusion:   Two-stream (RGB + Flow averaging)                 │
└─────────────────────────────────────────────────────────────┘
\\\

---

## Design Rationale

1. **1D Conv over LSTM for Text** (Gavrilyuk vs. Hu)
   - Simpler, faster, fewer parameters
   - Fixed-size output without sequential processing
   - Works well with small sentence vocabularies

2. **I3D over 2D CNN for Video** (Gavrilyuk vs. Hu)
   - Captures temporal dynamics (optical flow naturally)
   - Pre-trained on large video dataset (Kinetics)
   - Inflated 3D conv: proven effective for action understanding

3. **Dynamic Filters** (Novel to Gavrilyuk)
   - Text directly modulates visual processing
   - Adaptive to specific sentence query
   - Enables zero-shot generalization (OOV words via Word2Vec)

4. **Two-Stream Architecture**
   - Appearance (RGB) + Motion (Flow) are complementary
   - Proven in action recognition (Wang et al., 2016)
   - Independent streams allow flexible fusion strategies

---

**Last Updated**: June 7, 2026  
**Reference**: Gavrilyuk et al., CVPR 2018, Sections 3.1–3.3
