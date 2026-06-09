# Actor and Action Video Segmentation from a Sentence
**Gavrilyuk et al., CVPR 2018 — Deep Dive & Critical Analysis**

---

## 1. Bức tranh tổng thể

### Bài toán

Các công trình trước (Xu et al. A2D, Kalogeiton et al.) đều segment từ một **vocabulary cố định** gồm 43 actor-action pair. Giới hạn này có hai vấn đề:

- Không phân biệt được fine-grained instance: "người đứng bên trái" vs "người đứng bên phải" cùng là `adult_standing`
- Không generalize ra ngoài vocabulary: nếu xuất hiện actor/action mới, model bó tay

**Đóng góp cốt lõi của paper:** Thay vocabulary bằng **câu tiếng tự nhiên** làm query. Input là một câu mô tả + một video clip → output là pixel-level binary mask cho đúng actor đó đang thực hiện đúng action đó.

### Ba đóng góp chính

1. Định nghĩa task mới: *Actor and Action Segmentation from a Sentence*
2. Kiến trúc encoder-decoder end-to-end cho video (không phải ảnh đơn)
3. Dataset annotation: mở rộng A2D và J-HMDB với 7,500+ câu mô tả tiếng tự nhiên

---

## 2. Kiến trúc Model

```
Câu query ──► TextualEncoder ──► vector T [B, 300]
                                       │
                                       ▼
                              FC → tanh → Dynamic Filters
                                       │
Video ──► I3D ──► AvgPool(temporal) ──► Feature Maps [32×32, 128×128, 512×512]
                                       │
                                       ▼
                              Filter ⊗ Feature → Response Maps
                                       │
                                       ▼
                              Loss trên cả 3 resolution
                              (chỉ dùng S_512 lúc inference)
```

---

## 3. Textual Encoder

### Pipeline

```
Word2Vec (frozen, Google News, 300-dim)
    → pad câu về cùng độ dài
    → [B, L, 300]
    → transpose → [B, 300, L]
    → Conv1D(kernel=2, out=300)   # Bigram extractor
    → ReLU
    → Global Max Pooling
    → [B, 300]
```

### Tại sao CNN 1D thắng LSTM?

| | CNN 1D | LSTM |
|---|---|---|
| **Cơ chế** | Bắt n-gram cục bộ (bigram) | Chuỗi tuần tự, long-range dependency |
| **Phù hợp với** | "black dog", "running fast" — cụm danh/động ngữ liền kề | Câu dài, phụ thuộc xa |
| **Kết quả (IoU)** | 53.6% | 51.8% (vanilla) / 52.1% (Bi-LSTM) |

**Lý do thực tế:** Câu query trong dataset trung bình 7.3 từ, chứa nhiều cụm cục bộ có nghĩa mạnh. CNN 1D với kernel=2 hoạt động như một **Bigram Extractor**, Global Max Pooling giữ lại đặc trưng nổi bật nhất của toàn câu. LSTM bị loãng thông tin do phải xử lý tuần tự.

### Điểm yếu: Word2Vec Frozen

Paper không fine-tune embedding vì dataset quá nhỏ (~3,000 video train). Hậu quả: từ "rolling" của quả bóng và "rolling" của chú chó có **cùng một vector representation** — mất hoàn toàn domain-specific semantic trong ngữ cảnh video.

---

## 4. Video Encoder

### I3D — Inflated 3D ConvNet

- Inflate 2D filter của VGG/Inception lên 3D để xử lý spatio-temporal
- Pretrained: ImageNet (nhận dạng vật thể) + Kinetics (nhận dạng hành động)
- Lấy output tại **inception block trước last max-pooling** — đây là chi tiết quan trọng ảnh hưởng đến số channel

```
I3D(video N×512×512×3)
    → [B, 832, T', 32, 32]  (sau inception block cuối)
    → Average Pooling theo temporal
    → [B, 832, 32, 32]
    → L2-norm per spatial position
    → append spatial coords (x, y) normalized [-1, 1]
    → [B, 834, 32, 32]
```

### Spatial Coordinate Appending — "CoordConv trick"

Conv thông thường bị **translation invariant** — mù về vị trí tuyệt đối trong không gian. Bằng cách append 2 channel tọa độ (x, y) chuẩn hóa `[-1, 1]` vào feature map, model có thể học được spatial qualifier:

- "người đứng **bên trái**" → vùng x âm có activation cao hơn
- "con chó **phía trên**" → vùng y âm có activation cao hơn

Kỹ thuật này sau được formalize trong paper "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution" (2018).

### Ablation — Temporal Context

| N frames | Overall IoU |
|---|---|
| 1 | 48.2% |
| 4 | 52.2% |
| 8 | 52.8% |
| **16** | **53.6%** |

→ Temporal context đóng vai trò quan trọng. Chọn N=16 cho tất cả experiment.

---

## 5. Decoder — Dynamic Filters

### Ý tưởng cốt lõi

Thay vì dùng static filter (giống nhau cho mọi query), **sinh filter động** từ text representation. Filter này sau đó convolved với visual feature map để tạo response map.

### Pipeline chi tiết

```
T [B, 300]
    → FC → tanh → L2-norm
    → f_r [B, C_r]        # C_r = số channel của V_r

V_r [B, C_r, H, W]       # visual feature tại resolution r

S_r = Σ(f_r * V_r) theo dim channel  → [B, 1, H, W]
```

Đây thực chất là **channel-wise dot product** (1×1 convolution với filter động). Về bản chất toán học, đây là **Channel-wise Attention** — trả lời câu hỏi "kênh nào của visual feature map tương ứng với câu query này?"

### Ba resolution và Deconv flow

Paper dùng Deconv trên **visual feature trực tiếp**, không phải trên response map (khác với [Hu et al., 2016] và [Li et al., 2017]):

```
V_32 [B, 834, 32, 32]
    → Deconv(kernel=8×8, stride=4) → Conv(3×3) → ReLU
    → V_128 [B, 256, 128, 128]
    → append coords → [B, 258, 128, 128]
    → Deconv(kernel=8×8, stride=4) → Conv(3×3) → ReLU
    → V_512 [B, 128, 512, 512]
    → append coords → [B, 130, 512, 512]
```

**Lý do deconv trên feature thay vì response map:** Giữ lại thông tin không gian phong phú hơn, xử lý vật thể nhỏ tốt hơn, output mượt hơn.

---

## 6. Training

### Multi-Resolution Loss

$$L = \sum_{r \in \{32, 128, 512\}} \alpha_r \cdot L_r$$

$$L_r = \frac{1}{r^2} \sum_{i=1}^{r} \sum_{j=1}^{r} \log(1 + \exp(-S^r_{ij} \cdot Y^r_{ij}))$$

Đây là **logistic loss** (binary cross-entropy dạng margin). $Y^r_{ij} \in \{-1, +1\}$ (không phải {0, 1} như BCE thông thường).

**Tại sao multi-resolution loss quan trọng?**

| Setting | Overall IoU |
|---|---|
| Single resolution (512 only) | 49.4% |
| **Multi-resolution (32+128+512)** | **53.6%** |

Multi-resolution loss hoạt động như **skip-connection** về gradient — gradient flow từ resolution thấp về sâu hơn trong mạng, tránh vanishing gradient.

### Optimizer & Schedule

- Adam, lr=0.001
- Divide lr by 10 every 5,000 iterations
- Train 15,000 iterations total
- Chỉ finetune **last inception block** của I3D, phần còn lại frozen

---

## 7. Two-Stream: RGB + Optical Flow

### Fusion Strategy

Model RGB và Flow là **hai mạng riêng biệt** với cùng kiến trúc (không share weight). Fusion bằng weighted average của response map:

$$S_{final} = 2 \cdot S_{RGB} + 1 \cdot S_{Flow}$$

| Stream | Overall IoU |
|---|---|
| RGB only | 53.6% |
| Flow only | 49.5% |
| **RGB + Flow (2:1)** | **55.1%** |

Flow kém hơn RGB đáng kể — năm 2018 họ dùng TV-L1 optical flow, noisy và tốn compute. Với RAFT (2020) hoặc FlowFormer (2022), gap này chắc hẹp lại đáng kể.

---

## 8. Kết quả & Đánh giá

### A2D Sentences

| Model | mAP (0.5:0.95) | Overall IoU | Mean IoU |
|---|---|---|---|
| Hu et al. [6] | 2.0 | 21.3 | 12.8 |
| Li et al. [15] | 3.3 | 24.8 | 14.4 |
| Hu et al. [6]★ (finetuned) | 13.2 | 47.4 | 35.0 |
| Li et al. [15]★ (finetuned) | 16.3 | 51.5 | 35.4 |
| **This paper (RGB)** | **19.8** | **53.6** | **42.1** |
| **This paper (RGB+Flow)** | **21.5** | **55.1** | **42.6** |

### J-HMDB Sentences (zero-shot generalization — train trên A2D, test trên J-HMDB)

| Model | mAP | Overall IoU | Mean IoU |
|---|---|---|---|
| Hu et al. [6] | 17.8 | 54.6 | 52.8 |
| Li et al. [15] | 17.3 | 52.9 | 49.1 |
| **This paper** | **23.3** | **54.1** | **54.2** |

**Lưu ý quan trọng:** Improvement lớn nhất ở **Mean IoU** so với Overall IoU → model đặc biệt tốt hơn ở **small object segmentation** (Overall IoU bị bias về vật thể lớn).

---

## 9. Phân tích phản biện — Góc nhìn Reviewer

### Điểm mạnh thực sự

- End-to-end trainable từ text → pixel mask, không cần intermediate proposal
- Word2Vec pretrained generalize ra ngoài training vocabulary
- Multi-resolution loss là inductive bias rất thực tế

### Điểm yếu bị che khuất

**1. Evaluation chỉ trên middle frame**
Model được train và evaluate trên frame **giữa của mỗi clip**, không phải toàn bộ video. Điều này làm kết quả tốt hơn thực tế khi deploy trên streaming video thật sự — vì middle frame thường có actor rõ nét nhất.

**2. Dynamic filter = Channel-wise Attention, không phải spatial cross-attention**
1×1 conv chỉ trả lời "kênh nào quan trọng?", không capture được spatial relationship giữa các vùng trong ảnh và các từ trong câu. Câu "người đứng bên cạnh xe đạp" — "bên cạnh" là quan hệ spatial giữa hai entity, thứ mà 1×1 filter không model được.

Nếu làm vào 2022+, đây là chỗ cần Cross-Attention của Transformer (như LAVT, ReferFormer).

**3. Word2Vec frozen → mất domain semantic**
"rolling" của bóng và "rolling" của chó cùng một vector. Dataset quá nhỏ nên tác giả không dám finetune, nhưng đây là hạn chế về khả năng generalize sang domain mới.

**4. Annotation bias**
6,656 câu cho A2D được viết theo guideline từ annotator → phân phối ngữ pháp và từ vựng hạn chế. Query thực tế của người dùng đa dạng hơn rất nhiều.

**5. Flow stream dùng TV-L1 (2018)**
Noisy, tốn compute, nhạy cảm với camera motion. RAFT (2020) hay FlowFormer (2022) sẽ cải thiện đáng kể kết quả phần flow stream.

### Trajectory về sau

Paper này đặt nền móng cho một dòng task — *Referring Video Object Segmentation (RVOS)*. Sau 2018, các công trình kế thừa đã address đúng những điểm yếu trên:

| Paper | Cải tiến so với Gavrilyuk 2018 |
|---|---|
| CMSA (2020) | Cross-modal self-attention giữa text và visual |
| LAVT (CVPR 2022) | Language-Aware Visual Transformer — cross-attention ở từng layer |
| ReferFormer (CVPR 2022) | Object-level query, end-to-end video tracking |
| OnlineRefer (2023) | Online inference, không cần clip offline |

---

## 10. Code Implementation — Phân tích & Ghi chú

### Code của AI kia — Đúng và Sai chỗ nào

**Đúng:**
- Transpose trước Conv1D là chuẩn PyTorch
- `torch.sum(V * f, dim=1)` là cách đúng để implement channel-wise dot product
- Deconv apply trên visual feature, không phải response map
- append spatial coords trước mỗi resolution

**Cần lưu ý:**
- `nn.BCEWithLogitsLoss()` tương đương logistic loss với label {0,1}. Paper dùng label {-1, +1} — cần implement `log(1 + exp(-S*Y))` thủ công nếu muốn đúng hoàn toàn
- Số channel sau deconv cần match chính xác: V_32 có 832+2=834 channels, nhưng fc_filter_32 output cũng phải là 834 — code kia handle đúng
- L2-normalization nên apply **trước** khi append coords, không phải sau

### Implementation Order đề xuất

```
1. TextualEncoder          ← standalone, test ngay được
2. SpatialCoordAppender    ← utility function
3. I3DFeatureExtractor     ← wrap pretrained I3D, extract tại Mixed_5c
4. DynamicFilterDecoder    ← core module
5. MultiResolutionLoss     ← custom loss
6. ActorActionModel        ← assemble tất cả
7. DataLoader              ← A2D Sentences dataset
8. Training loop           ← Adam + lr schedule
```

### I3D — Cách lấy feature tại đúng layer

```python
# Dùng repo: https://github.com/piergiaj/pytorch-i3d
# Load pretrained, sau đó hook vào Mixed_5c

from i3d import InceptionI3d

model = InceptionI3d(400, in_channels=3)
model.load_state_dict(torch.load('rgb_imagenet.pt'))

# Lấy output trước last max-pooling
# Mixed_5c output: [B, 1024, T', 7, 7] với input 224×224
# Với input 512×512, spatial dim sẽ là 32×32
```

---

## 11. Key Takeaways

1. **Dynamic Filter = Language-conditioned Channel-wise Attention** — ý tưởng hay nhưng limited về spatial reasoning

2. **Multi-resolution loss = implicit skip connection** — gradient flow về sâu hơn, cải thiện 4.2% IoU tuyệt đối

3. **Temporal context quan trọng** — N=16 frame thắng N=1 tới 5.4% IoU, xác nhận video >> image đơn cho task này

4. **Evaluation protocol cần đọc kỹ** — "middle frame only" làm số đẹp hơn thực tế

5. **Paper này là foundation của RVOS track** — hiểu kỹ paper này là nền để đọc LAVT, ReferFormer, OnlineRefer

---

*Tổng hợp từ paper gốc (CVPR 2018), transcript thảo luận và phân tích chéo.*
