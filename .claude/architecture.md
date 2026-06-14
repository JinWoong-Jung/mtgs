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

각 `InteractionBlock`은 DINOv2 ViT 블록 3개를 감싸며, **inject_first (고정)** 순서로 실행:

```
[Injector]            person → scene cross-attn
[ViT blocks 3개]      scene self-attn (DINOv2)
[Extractor]           scene → person cross-attn
[People Interaction]  person token 간 self-attention
[People Temporal]     temporal self-attention
```

4단계 완료 후 → `img_layers[0..3]`, `gaze_layers[0..3]` 수집

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

## Step 5: Social Gaze (GazeGraphBlock)

```python
# gaze_projs로 4 stage trunk token projection+concat
proj_tokens = cat([gaze_projs[i](gaze_layer) for i in range(4)])  # (B*T, N, 512)

# in-out 분류
inout = inout_decoder(proj_tokens.view(B*T*N, -1))   # (B*T*N, 1)

# GazeGraphBlock 적용 (단 1회)
lah, laeo, coatt, null_in_probs, null_out_probs, _ = gaze_graph_block(
    proj_tokens, gaze_vec, head_bboxes, gaze_hm, inout_logit
)
```

**GazeGraphBlock 내부 흐름:**

```
[Step 1a] Source node enrichment: P'_i = LN(P_i + XAttn(P_i, pool(H_i)))  ← heatmap 8×8 grid
[Step 1b] Target node pre-init: tgt_j = h_j + gate_j · MLP([h_j ‖ overlap_weighted_msg_j])
            gate_j = max_i(overlap(H_i, b_j));  msg_j = softmax_overlap weighted sum of h_i

[Step 2] Edge init (type별 1D scalar prior):
           e(i→Pj)      : prior = cos(g_i, dir_ij)           ← gaze-target cosine
           e(i→null_in) : prior = 1 - Σ_j overlap(H_i, b_j) ← non-person heatmap mass
           e(i→null_out): prior = 1 - sigmoid(inout_logit_i) ← out-of-frame prob
           E_init = MLP([v_src; v_tgt; W_scalar(prior); type_emb]) × ev

[Step 3] _UnifiedRefiner × L(=2) layers:
           Row-attn   (source i가 자신의 N+2 outgoing edges attend)
         → Col-attn   (target k가 N incoming sources attend; null_out 제외)
         → Edge refresh: LN(E + MLP([E ‖ row_ctx ‖ col_ctx]))
         → Node update (attention pooling, source/target 별도 MLP):
             v_src_i ← out_agg 기반,  v_tgt_j ← in_agg 기반
         → Node re-injection: LN(E + MLP([E ‖ v_src ‖ v_tgt]))

[Step 4] Readout:
           LAH      : head_lah(E[i→j])                           — directed edge 단독
           LAEO     : head_laeo([E[i→j]; E[j→i]]) symmetrized    — 전용 MLP
           SA       : head_sa([ni_i ‖ ni_j ‖ |diff| ‖ E[i→j] ‖ E[j→i]]) symmetrized
                        ni = E[·→null_in] (장면 응시 패턴), 5·De
           null_in  : head_null_in(E[i→null_in])
           null_out : head_null_out(E[i→null_out])
```

수식 상세는 [gaze_graph_math.md](gaze_graph_math.md) 참조.

---

## 핵심 모듈 파일 위치

| 모듈 | 파일 | 클래스/함수 |
|------|------|------------|
| 전체 MTGS 네트워크 | `mtgs/networks/mtgs_net.py` | `MTGS` |
| Lightning 학습 모델 | `mtgs/networks/models.py` | `MTGSModel` |
| Gaze 인코더 | `mtgs/networks/mtgs_net.py` | `GazeEncoder` |
| ViT-Adaptor 블록 | `mtgs/networks/adaptor_modules.py` | `InteractionBlock` |
| Unified Gaze Graph | `mtgs/networks/adaptor_modules.py` | `GazeGraphBlock` |
| DPT 히트맵 디코더 | `mtgs/networks/mtgs_net.py` | `ConditionalDPTDecoder` |
| InOut 디코더 | `mtgs/networks/mtgs_net.py` | `InOutDecoder` |

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
  + lambda_null * (loss_null_in + loss_null_out)   # dual-null aux loss (lambda_null=0.5)
)
```

UCO-LAEO, VideoCoAtt 데이터에는 heatmap/angular 손실에 0.1 가중치 적용 (gaze GT 품질이 낮기 때문).

---

## 학습 최적화 세부사항

- **옵티마이저**: AdamW (weight_decay=1e-3)
- **학습률 (4 param groups)**:
  - `gaze_encoder_temporal` — base × 3
  - `people_temporal` — base × 3
  - `gaze_graph_block` — base × 3
  - 나머지 (base, 기본 1e-6)
- **스케줄러**: CosineAnnealingWarmRestarts (T_0=20 epochs, 4 epoch warmup)
- **SWA**: epoch 12부터 시작, 6 epoch annealing
- **정밀도**: bf16-mixed
- **Gradient Clipping**: 사용 안 함 (`gradient_clip_val: null`)
- **Frozen 레이어**: DINOv2 encoder + image_tokenizer (기본값)
- **`gaze_graph.frozen=true` 시**: trunk 전체 freeze + `gaze_graph_block` 파라미터만 학습

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
