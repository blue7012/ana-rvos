# Handoff Document — Actor and Action Video Segmentation from a Sentence
## Reimplementation for Benchmarking (Gavrilyuk et al., CVPR 2018)

---

## 1. Project Goal

Reimplement the model from **"Actor and Action Video Segmentation from a Sentence"** (Gavrilyuk, Ghodrati, Li, Snoek — CVPR 2018) in PyTorch for benchmarking purposes. The model takes a natural language sentence query and a video clip as inputs, and produces a pixel-level segmentation mask identifying the actor and action described by the query.

**Environment:** Google Colab (GPU recommended), Python 3.12, PyTorch.

**Dataset in use:** YouTube-VOS 2019, stored at:
```
/content/drive/MyDrive/att-3/ana-rvos/data/youtube-vos-2019/train/JPEGImages/<video_id>/
```
Each video folder contains JPEG frames named `00000.jpg, 00005.jpg, ..., 00095.jpg` (20 frames, spaced by 5).

---

## 2. Paper Architecture — Quick Reference

The model has three components wired together. Read this before touching any code.

```
TEXTUAL ENCODER
  Sentence string
    → Word2Vec (300-dim, Google News, frozen)       shape: (max_len, 300)
    → 1D Conv (kernel=2, out=300) + ReLU
    → Global max-pool
    → Sentence vector                               shape: (B, 300)
    → [split into 3 FC branches for decoder]

VIDEO ENCODER
  N frames (N=16, 512×512×3)
    → I3D backbone (pre-trained Kinetics RGB)
    → Stop at Mixed_4f                              shape: (B, 832, T'≈4, 32, 32)
    → Avg-pool over temporal dim                    shape: (B, 832, 32, 32)
    → L2-normalise per spatial position
    → Append (y, x) coordinate channels            shape: (B, 834, 32, 32)

DECODER  ← NOT YET BUILT
  Sentence vector (B, 300)
    → FC layer → dynamic filter (1×1×832)   for 32×32 video feature
    → FC layer → dynamic filter (1×1×256)   for 128×128 video feature
    → FC layer → dynamic filter (1×1×128)   for 512×512 video feature
  Convolution: filter * video_feature = response map
    → 32×32 response map
    → Deconv → 128×128 response map
    → Deconv → 512×512 response map   ← final segmentation mask
  Loss: multi-resolution logistic loss (α weighted sum over 3 scales)
```

**Key numbers to remember:**
- Sentence vector: `(B, 300)`
- Video feature map: `(B, 834, 32, 32)` — 832 visual + 2 coordinate channels
- Dynamic filters: `(1×1×832)`, `(1×1×256)`, `(1×1×128)`
- Response maps: `32×32`, `128×128`, `512×512`
- Loss resolutions: `R = {32, 128, 512}` with weights `α_r = 1` for all

---

## 3. What Has Been Built

### 3.1 Textual Encoder — `textual_encoder.py` ✅ COMPLETE

**Status:** Working. Produces a `(B, 300)` sentence vector ready for the decoder's FC layers.

**File structure (7 Colab cells):**

| Cell | Function | What it does |
|------|----------|--------------|
| 1 | Setup | `!pip install gensim` |
| 2 | Imports | torch, gensim, numpy |
| 3 | `load_word2vec()` | Downloads Google News 300-dim vectors (~1.5 GB, one-time) |
| 4 | `sentence_to_embedding()` | Sentence string → padded `(max_len, 300)` tensor |
| 5 | `TextualEncoder` | 1D-CNN module |
| 6 | `encode_sentence()` | Convenience wrapper for inference |
| 7 | Sanity check | Tests with paper's example queries |

**`TextualEncoder` module internals:**
```python
forward(x: (B, max_len, 300)) → (B, 300)

  x = x.permute(0, 2, 1)      # (B, 300, max_len)
  x = Conv1d(300, 300, k=2)   # (B, 300, max_len-1)
  x = ReLU(x)
  x = x.max(dim=-1).values    # (B, 300)  ← global max-pool
```

**Important implementation details:**
- `max_len=20`: sentences are padded to this fixed length so batches can be stacked. The network doesn't care about padding because max-pool always picks the highest activation (padding zeros → near-zero activations after ReLU, never win the max).
- `in_channels=300` in Conv1d = depth of each word column (300 word2vec dims), NOT the number of words.
- `kernel_size=2` = the filter looks at 2 adjacent words at once and slides one word at a time.
- Word embeddings are **frozen** — only the Conv1d weights (180,300 params) are learned.

---

### 3.2 Video Encoder — `video_encoder_fix2.py` ✅ COMPLETE

**Status:** Working. Produces a `(B, 834, 32, 32)` feature map ready for the decoder's dynamic filter convolution.

**File structure (6 Colab cells):**

| Cell | Function | What it does |
|------|----------|--------------|
| 1 | Setup | `git lfs install && git clone pytorch-i3d` |
| 2 | Imports | torch, PIL, matplotlib, InceptionI3d |
| 3 | `load_i3d()` | Loads full pre-trained I3D (no `final_endpoint`) |
| DIAGNOSTIC | Debug cell | Verifies param count + prints module names |
| 4 | `load_video_frames()` | Loads N JPEG frames from folder, normalises to [-1,1] |
| 5 | `VideoEncoder` | Hook-based wrapper around I3D |
| 6 | Sanity check + visualisation | Forward pass + matplotlib output |

**`VideoEncoder` module internals:**
```python
forward(x: (B, 3, 16, 512, 512)) → (B, 834, 32, 32)

  _ = self.i3d(x)                      # full I3D forward
  feat = self._feat['mixed_4f']        # hook captured (B, 832, 4, 32, 32)
  feat = feat.mean(dim=2)              # avg-pool T'=4 → (B, 832, 32, 32)
  feat = F.normalize(feat, p=2, dim=1) # L2-norm per spatial position
  feat = _append_spatial_coords(feat)  # append y,x grids → (B, 834, 32, 32)
```

**Why `Mixed_4f`:** With 512×512 input, I3D applies 4 spatial halvings before Mixed_4f (stride-2 conv + 3× MaxPool): 512→256→128→64→32. Mixed_4f has exactly 832 output channels, matching the decoder's dynamic filter dimensions in Figure 2.

**I3D backbone:** `piergiaj/pytorch-i3d`, RGB stream, pre-trained on ImageNet + Kinetics-400. Weights at `/content/pytorch-i3d/models/rgb_imagenet.pt` (~28 MB). **12,697,264 parameters.**

---

## 4. Bugs Hit and How We Fixed Them

This section is critical context — all three bugs will waste hours if you hit them again.

### Bug 1 — `InceptionI3d` with `final_endpoint='Mixed_4f'` → 0 params

**Symptom:** `Params (up to Mixed_4f): 0`

**Root cause:** The pytorch-i3d constructor pattern is:
```python
if self._final_endpoint == 'Mixed_4f': return   # ← bails out FIRST
self.Mixed_4f = InceptionModule(...)             # ← never reached
```
The check fires before the layer is created, so the model returns with nothing built.

**Fix:** Load the full model with the default endpoint, then stop the forward pass manually.
```python
# WRONG
i3d = InceptionI3d(num_classes=400, in_channels=3, final_endpoint='Mixed_4f')

# CORRECT
i3d = InceptionI3d(num_classes=400, in_channels=3)   # default = 'Logits'
```

---

### Bug 2 — `AttributeError: 'InceptionI3d' has no attribute 'conv3d_1a_7x7'`

**Symptom:** Manual layer traversal in `_extract_mixed4f()` fails on the first call.

**Root cause:** The pytorch-i3d version cloned uses **CamelCase** attribute names (`Mixed_4f`, `Conv3d_1a_7x7`) while the code assumed lowercase (`mixed_4f`, `conv3d_1a_7x7`).

**Confirmed by running:**
```python
[n for n, _ in i3d_backbone.named_children()]
# → ['avg_pool', 'dropout', 'logits', 'Conv3d_1a_7x7', ..., 'Mixed_4f', ...]
```

**Fix:** Switched from hardcoded attribute access to a **hook-based approach** with automatic name discovery. `VideoEncoder.__init__` now searches `named_children()` for any module whose name matches `mixed_4f` or `Mixed_4f` (case-insensitive), then attaches a `register_forward_hook` to capture its output. No attribute names hardcoded anywhere.

---

### Bug 3 — Conv1D parameter confusion (conceptual, not code)

**Symptom:** Confusion about what `in_channels=300`, `out_channels=300`, `kernel_size=2` mean in the context of word embeddings.

**Clarification:**
- `in_channels=300` = the **depth** of each position in the sequence (300 word2vec dimensions per word), not the number of words
- `kernel_size=2` = the filter slides over **2 adjacent words** at a time
- The output length shrinks by 1 (`seq_len - 2 + 1`) because the kernel can't extend past the edge — this doesn't matter since max-pool collapses the time axis anyway
- `max_len` padding exists only for **batching**, not for the network's math

---

## 5. Current State

```
Textual Encoder   ✅  textual_encoder.py       (B, 300) sentence vector
Video Encoder     ✅  video_encoder_fix2.py    (B, 834, 32, 32) feature map
─────────────────────────────────────────────────────────────────────────────
Decoder           ❌  not started              dynamic filters + deconv
Full model        ❌  not started              encoder-decoder integration
Training loop     ❌  not started              multi-resolution logistic loss
Evaluation        ❌  not started              IoU metrics on A2D/J-HMDB
```

The two encoders have been verified independently with sanity checks. They have not yet been wired together end-to-end.

---

## 6. Next Immediate Steps

### Step 1 — Build the Decoder (`decoder.py`)

This is the next file to write. The decoder takes the sentence vector `(B, 300)` from the Textual Encoder and the video feature map `(B, 834, 32, 32)` from the Video Encoder and produces three response maps.

**Three FC branches (one per resolution):**

```python
# FC layer generates a dynamic filter for each resolution
# Input: sentence vector (B, 300)
# Output: filter weights that get convolved with video features

filter_32   = tanh(W_32   @ sentence_vec + b_32)   # shape: (B, 832)  for 32×32 map
filter_128  = tanh(W_128  @ sentence_vec + b_128)   # shape: (B, 256)  for 128×128 map
filter_512  = tanh(W_512  @ sentence_vec + b_512)   # shape: (B, 128)  for 512×512 map
```

**Note:** The 128 and 512 channel video features at those resolutions come from the deconvolutional network applied to the video feature map — not directly from I3D. The decoder architecture is:
1. Dynamic filter convolution at 32×32 → response map 32×32
2. Deconv the **video feature** (not the response map) → 128×128 feature (256 ch)
3. Dynamic filter convolution at 128×128 → response map 128×128
4. Deconv → 512×512 feature (128 ch)
5. Dynamic filter convolution at 512×512 → response map 512×512 (final output)

**Deconv block details (from paper Sec 3.3):**
```python
# Each deconv block:
DeconvTranspose2d(kernel=8×8, stride=4)   # upsample ×4
Conv2d(kernel=3×3, stride=1)              # refine
```

### Step 2 — Wire the Full Model

```python
class ActorActionSegmenter(nn.Module):
    def __init__(self):
        self.text_encoder  = TextualEncoder(...)
        self.video_encoder = VideoEncoder(...)
        self.decoder       = Decoder(...)

    def forward(self, video, sentence_embedding):
        text_feat  = self.text_encoder(sentence_embedding)   # (B, 300)
        video_feat = self.video_encoder(video)               # (B, 834, 32, 32)
        masks      = self.decoder(text_feat, video_feat)     # dict of 3 response maps
        return masks
```

### Step 3 — Implement the Multi-resolution Loss

From paper Section 3.4:
```python
# Total loss = weighted sum across resolutions R = {32, 128, 512}
L = sum(alpha_r * L_r for r in [32, 128, 512])   # alpha_r = 1 for all

# Per-resolution loss = mean logistic loss over pixels
L_r_ij = log(1 + exp(-S_r_ij * Y_r_ij))
# S_r_ij = response map value at pixel (i,j) for resolution r
# Y_r_ij = binary ground truth label (+1 actor/action, -1 background)
```

Ground truth masks need to be downsampled to 32×32 and 128×128 for the lower-resolution losses.

### Step 4 — Training Setup

From paper Section 3.4:
- Optimiser: Adam, `lr=0.001`
- LR schedule: divide by 10 every 5,000 iterations
- Total iterations: 15,000
- Finetuned layers: last inception block of video encoder + full decoder
- Frozen layers: all I3D layers before Mixed_4f

---

## 7. File Index

| File | Purpose | Status |
|------|---------|--------|
| `textual_encoder.py` | Cells 1–7: Word2Vec loading, `sentence_to_embedding`, `TextualEncoder`, `encode_sentence`, sanity check | ✅ Use as-is |
| `video_encoder.py` | Original video encoder — has bugs (do not use) | ⚠️ Superseded |
| `video_encoder_fix.py` | Fixed Cell 3 + Cell 5 (manual layer traversal) — still has CamelCase naming bug | ⚠️ Superseded |
| `video_encoder_fix2.py` | Final working version: diagnostic cell + hook-based `VideoEncoder` | ✅ Use as-is |
| `decoder.py` | Dynamic filter generation + deconvolutional upsampling | ❌ Not yet written |

**To reconstruct the working notebook, run cells in this order:**
1. `textual_encoder.py` Cells 1–6 (skip Cell 7 sanity check if short on time)
2. `video_encoder_fix2.py` Cell 1 (setup) → Cell 2 (imports) → Cell 3 (load_i3d) → DIAGNOSTIC → Cell 4 (load_video_frames) → Cell 5 (VideoEncoder) → Cell 6 (sanity check)

---

## 8. Key References

| Item | Detail |
|------|--------|
| Paper | Gavrilyuk et al., CVPR 2018 — "Actor and Action Video Segmentation from a Sentence" |
| I3D backbone | `github.com/piergiaj/pytorch-i3d` — RGB stream, `rgb_imagenet.pt` |
| I3D paper | Carreira & Zisserman, CVPR 2017 — "Quo Vadis, Action Recognition?" |
| Word2Vec | Google News 300-dim, loaded via `gensim.downloader.api.load("word2vec-google-news-300")` |
| Dataset | A2D Sentences + J-HMDB Sentences (paper's evaluation sets) |
| Your data | YouTube-VOS 2019 (currently used only for sanity-checking the pipeline) |
