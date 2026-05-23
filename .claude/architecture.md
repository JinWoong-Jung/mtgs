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
[People Interaction] person_tokens self-attention (사람들 간 상호작용)
[People Temporal]   person_tokens temporal self-attention (시간 축 상호작용)
```

총 4개 InteractionBlock → `img_layers[0..3]`, `gaze_layers[0..3]` 수집

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

## Step 5: Social Gaze 디코딩

**두 모드 공통** — `gaze_projs`와 `inout_decoder`는 transformer/graph 모드 모두 공유.  
Transformer 체크포인트에서 Graph 모드로 fine-tuning 시 이 가중치들이 재사용됨.

```python
# 모든 InteractionBlock에서 나온 person token을 projection 후 concat (양 모드 공통)
proj_tokens = cat([gaze_projs[i](gaze_layer) for i, gaze_layer in enumerate(gaze_layers)])
# shape: (B*T, N, 128*4=512)

# In-out 분류 (양 모드 공통 decoder)
inout = inout_decoder(proj_tokens.view(B*T*N, -1))     # (B*T*N, 1)
```

**Transformer 모드 전용** — pair-wise MLP decoder:

```python
# 모든 pair 조합 생성 (permutation)
indices = permutations(range(N), 2)    # (N*(N-1), 2)
pairs = cat([proj_tokens[:, i], proj_tokens[:, j]], dim=-1)   # (B*T*num_pairs, 1024)

lah   = decoder_lah(pairs)   # (B*T, num_pairs)
coatt = decoder_sa(pairs)    # (B*T, num_pairs)

# LAEO = min(LAH(i→j), LAH(j→i))
laeo[pi] = min(lah[pi], lah[corr_idx])
```

**Graph 모드 전용** — `SocialGraphBlock`의 edge logit이 직접 LAH/SA 예측:

```python
# lah_from_graph = aux_lah_logits[-1]  → 마지막 블록의 LAH edge logit (B*T, N*(N-1))
# sa_from_graph                        → 마지막 블록의 SA edge logit
# LAEO: min(lah_ij, lah_ji) — _reverse_edge_idx() 로 역방향 인덱스 계산
```

---

## 핵심 모듈 파일 위치

| 모듈 | 파일 | 클래스/함수 |
|------|------|------------|
| 전체 MTGS 네트워크 | `mtgs/networks/mtgs_net.py` | `MTGS` |
| Lightning 학습 모델 | `mtgs/networks/models.py` | `MTGSModel` |
| Gaze 인코더 | `mtgs/networks/mtgs_net.py` | `GazeEncoder` |
| ViT-Adaptor 블록 | `mtgs/networks/adaptor_modules.py` | `InteractionBlock` |
| Graph Interaction | `mtgs/networks/adaptor_modules.py` | `SocialGraphBlock`, `TemporalGraphBlock` |
| DPT 히트맵 디코더 | `mtgs/networks/mtgs_net.py` | `ConditionalDPTDecoder` |
| Social 디코더 (transformer) | `mtgs/networks/mtgs_net.py` | `LinearDecoderSocialGraph` |
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
- **학습률**: 기본 1e-6, gaze_encoder_temporal 3×, social/temporal graph blocks 10×/5× (별도 param group)
- **스케줄러**: CosineAnnealingWarmRestarts (T_0=20 epochs, 4 epoch warmup)
- **SWA**: epoch 12부터 시작, 6 epoch annealing, lr=[1e-6, 1e-6, 1e-6, 3e-7] (4 param groups, 양 모드 동일)
- **정밀도**: bf16-mixed
- **Gradient Accumulation**: 1 (기본)
- **Gradient Clipping**: 사용 안 함 (`gradient_clip_val: null`) — clipping이 transformer 모드 수렴을 방해함
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
# val/test 각 3종 × 2 = 12개 메트릭, 모두 CPU 누적
self.val_coatt_auc  = tm.AUROC(task="binary", ignore_index=-1).cpu()
self.val_laeo_auc   = tm.AUROC(task="binary", ignore_index=-1).cpu()
self.val_lah_auc    = tm.AUROC(task="binary", ignore_index=-1).cpu()
# ... (test_ 버전 동일)
```

`.cpu()` 이유: 43k+ test 샘플을 GPU에 누적하면 스텝마다 growing tensor에 concat하는 비용이 증가해 속도가 급격히 저하됨 → CPU 누적으로 GPU 메모리 압박 해소.  
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
