# MTGS 모델 아키텍처 상세

## 전체 Forward 흐름

```
입력:
  x["image"]       : (B, T, C, H, W)         — 원본 프레임
  x["heads"]       : (B, T, N, C, H_h, W_h)  — 각 사람의 head crop (224×224)
  x["head_bboxes"] : (B, T, N, 4)             — 정규화된 head bounding box
  x["num_valid_people"] : scalar              — 실제 검출된 사람 수

B = batch, T = temporal window (2*temporal_context+1), N = num_people (padding 포함)
```

---

## Step 1: Gaze Token 인코딩

```python
# GazeEncoder.forward_backbone
gaze_emb = ResNet18(heads) → (B*T, N, 512)

# Temporal self-attention (T > 1일 때만)
gaze_emb += temporal_position_embedding   # learnable (T, 512)
gaze_emb = TransformerBlock(gaze_emb)     # (B*N, T, 512)

# GazeEncoder.forward_head
gaze_token = Linear(gaze_emb) + Linear(head_bbox)  # (B*T, N, 768)
gaze_vec   = normalize(Linear(gaze_emb))            # (B*T, N, 2) — 시선 방향 단위벡터
```

---

## Step 2: Scene Tokenization

```python
# DINOv2 ViT-B/14 (frozen)
image_tokens = dinov2.prepare_tokens_with_masks(image)  # (B*T, num_patches+1, 768)
```

---

## Step 3: ViT-Adaptor (4회 반복)

각 `InteractionBlock`은 DINOv2 ViT 블록 3개를 감싸 아래를 순서대로 수행:

```
[Injector]       person_tokens → image_tokens (cross-attn: person attends to scene)
[ViT blocks 3개]  image_tokens self-attention (DINOv2 레이어 3개)
[Extractor]      image_tokens → person_tokens (cross-attn: person reads updated scene)
```

이후 모드에 따라 분기:

**Transformer 모드:**
```
[People Interaction]  person_tokens self-attention (사람들 간 상호작용)
[People Temporal]     person_tokens temporal self-attention (시간 축 상호작용)
```

**Graph 모드:**
```
[SocialGraphBlock]    outgoing directed graph message passing (사람 간 spatial 상호작용)
[TemporalGraphBlock]  per-person MHA over T frames (시간 축 상호작용)
```

총 4개 반복 → `img_layers[0..3]`, `gaze_layers[0..3]` 수집

---

## Step 4: Heatmap 디코딩 (ConditionalDPTDecoder)

DPT(Dense Prediction Transformer) 스타일의 multi-scale 디코딩:

```
img_layers를 공간적 feature map으로 reshape: (B*T, D, H/patch, W/patch)

각 scale에서:
  Reassemble block: 공간 해상도 조정 (factor 32→16→8→4)
  gaze projection: person token → feature와 element-wise 곱 (conditional)
  FusionBlock: ResidualConvUnit × 2 + upsampling × 2

최종 head (conv layers) → (B*T*N, 1, H/2, W/2) → bilinear → (B*T*N, 1, 64, 64)
→ (B, T, N, 64, 64)
```

**Conditional Heatmap 생성 핵심**: `torch.einsum("bdhw,bnd->bndhw", f, g)` — scene feature와 person token의 element-wise 내적으로 사람별 조건부 히트맵 생성

---

## Step 5: Social Gaze 디코딩 (양 모드 공통)

`gaze_projs`, `inout_decoder`, `decoder_lah`, `decoder_sa`는 transformer/graph 모드 모두 공유.  
Transformer 체크포인트에서 Graph 모드로 fine-tuning 시 이 가중치들이 warm-start로 재사용됨.

```python
# 모든 InteractionBlock에서 나온 person token을 projection 후 concat
proj_tokens = cat([gaze_projs[i](gaze_layer) for i, gaze_layer in enumerate(gaze_layers)])
# shape: (B*T, N, 128*4=512)

# In-out 분류
inout = inout_decoder(proj_tokens.view(B*T*N, -1))     # (B*T*N, 1)

# Pair-wise social gaze 예측 (양 모드 동일)
indices = permutations(range(N), 2)                    # N*(N-1)개
pairs = cat([proj_tokens[:, i], proj_tokens[:, j]], dim=-1)   # (B*T*num_pairs, 1024)

lah   = decoder_lah(pairs).view(B*T, num_pairs)        # LAH
coatt = decoder_sa(pairs)                              # SA

# LAEO = min(LAH(i→j), LAH(j→i))
laeo[pi] = min(lah[pi], lah[corr_idx])
```

---

## 핵심 모듈 파일 위치

| 모듈 | 파일 | 클래스/함수 |
|------|------|------------|
| 전체 MTGS 네트워크 | `mtgs/networks/mtgs_net.py` | `MTGS` |
| Lightning 학습 모델 | `mtgs/networks/models.py` | `MTGSModel` |
| Gaze 인코더 | `mtgs/networks/mtgs_net.py` | `GazeEncoder` |
| ViT-Adaptor 블록 | `mtgs/networks/adaptor_modules.py` | `InteractionBlock` |
| Social Graph (node update) | `mtgs/networks/adaptor_modules.py` | `SocialGraphBlock` |
| Temporal Graph | `mtgs/networks/adaptor_modules.py` | `TemporalGraphBlock` |
| DPT 히트맵 디코더 | `mtgs/networks/mtgs_net.py` | `ConditionalDPTDecoder` |
| Social 디코더 (양 모드 공유) | `mtgs/networks/mtgs_net.py` | `LinearDecoderSocialGraph` |
| InOut 디코더 (양 모드 공유) | `mtgs/networks/mtgs_net.py` | `InOutDecoder` |

---

## 손실 함수 구성 (losses.py)

```python
total_loss = (
    3   * angular_loss    # 시선 방향 코사인 손실
  + 100 * dist_loss       # 시선 포인트 L2 거리 (heatmap mode에서는 0)
  + 1000 * heatmap_loss   # 히트맵 MSE
  + 2   * inout_loss      # in/out BCE
  + 1   * lah_loss        # LAH BCE (pos_weight=3)
  + 1   * laeo_loss       # LAEO BCE
  + 1   * coatt_loss      # SA BCE
)
```

UCO-LAEO, VideoCoAtt 데이터에는 heatmap/angular 손실에 0.1 가중치 적용 (gaze GT 품질이 낮기 때문).

---

## 학습 최적화 세부사항

- **옵티마이저**: AdamW (weight_decay=1e-3)
- **학습률 (transformer 모드)**: 기본 1e-6, gaze_encoder_temporal/people_temporal/decoder_sa 3×
- **학습률 (graph 모드)**: 기본 1e-6, social_graph_blocks 10×, temporal_graph_blocks 5×, gaze_encoder_temporal/decoder_lah/decoder_sa 3×
- **스케줄러**: CosineAnnealingWarmRestarts (T_0=20 epochs, 4 epoch warmup)
- **SWA**: epoch 12부터 시작, 6 epoch annealing
- **정밀도**: bf16-mixed
- **Gradient Clipping**: 사용 안 함 (`gradient_clip_val: null`)
- **Frozen 레이어**: DINOv2 encoder + image_tokenizer (기본값)

---

## 추론 시 중앙 프레임 사용

Validation/Test에서는 temporal window의 **중앙 프레임만** 평가에 사용:
```python
middle_frame_idx = int(t / 2)
gaze_hm_pred = gaze_hm_pred[:, middle_frame_idx, :]
```

---

## 평가 메트릭 (models.py)

Social gaze 메트릭은 AUROC + AveragePrecision (threshold-free, ranking 기반):

```python
# val/test 각 3종 × 2 = 12개 메트릭, 모두 compute_on_cpu=True
self.val_coatt_auc  = tm.AUROC(task="binary", ignore_index=-1, compute_on_cpu=True)
self.val_laeo_auc   = tm.AUROC(task="binary", ignore_index=-1, compute_on_cpu=True)
self.val_lah_auc    = tm.AUROC(task="binary", ignore_index=-1, compute_on_cpu=True)
# ... (test_ 버전 동일)
```

ignore_index=-1: 패딩된 pair label(-1)은 메트릭 계산에서 자동 제외.

---

## Test 배치 처리 (collate.py + vsgaze.py)

test 시 `num_people="all"`로 샘플마다 N이 다름 → 기본 collate 불가.

```
pad_collate_fn (mtgs/train/collate.py):
  N-dim 텐서 → max_N으로 zero-padding (heads, bboxes 등)
  pair-dim 텐서 → max_pairs로 -1 padding (lah_labels, coatt_labels 등)
  string 키 (path, dataset) → list로 수집

_filter_by_max_people (mtgs/datasets/vsgaze.py):
  test_dataset 구성 전에 N > max_people 샘플을 Subset으로 제거
  → batch 내 max_N 상한 제어 → VRAM 변동성 억제
```
