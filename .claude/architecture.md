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

각 `InteractionBlock`은 DINOv2 ViT 블록 3개를 감싸며, `interaction.order` 설정에 따라 실행 순서가 다름:

**`inject_first` (기본/원본):**
```
[Injector]       person → scene cross-attn
[ViT blocks 3개]  scene self-attn (DINOv2)
[Extractor]      scene → person cross-attn
[Social Block]   모드별 분기 (아래 참조)
```

**`extract_first` (신규):**
```
[Extractor]      scene → person cross-attn  ← scene-enriched 상태로 social 진행
[Social Block]   모드별 분기 (아래 참조)
[Injector]       person → scene cross-attn  ← socially-aware person token 주입
[ViT blocks 3개]  scene self-attn (DINOv2)
```

`InteractionBlock`에 `forward_extract_only()`, `forward_inject_vit()` 메서드 추가로 분리 호출 지원.

**Social Block 모드별 분기:**

**Transformer 모드:**
```
[People Interaction]  person_tokens self-attention
[People Temporal]     temporal self-attention
```

**Graph 모드 (2-graph 분리):**
```
trunk (LAH/LAEO/heatmap/inout):
  [SocialGraphBlock]          outgoing directed graph, softmax + dual-null (null_in/null_out)
  [TemporalGraphBlock]        per-person MHA over T frames
SA 분기 (CoAtt 전용, trunk로 되먹이지 않는 평행 read-out):
  [UndirectedSocialGraphBlock] sigmoid gated-mean, 대칭 edge, SA gaze prior
  [sa_temporal_blocks]         per-person MHA over T frames
```
- 두 그래프는 매 블록에서 **같은 입력 토큰**(`sa_in` = directed 블록 직전 trunk)으로부터 평행 처리.
- directed 블록만 trunk를 갱신 → 다음 블록·heatmap·scene으로 전파.
- undirected 블록은 자기 스냅샷(`sa_layers`)만 `decoder_sa`로 보냄.
- directed 블록은 LAH 방향 prior만 사용(`use_sa_prior=False`); SA gaze prior는 undirected 블록 소관.

**Hypergraph 모드:**
```
[HypergraphBlock]     person hyperedge N개 + null_in + null_out
                      → returns (h, attn_agg, attn_null_in)
                        attn_agg[i,j]  = "i가 j를 보는 정도" (N→E softmax person 기여분)
                        attn_null_in[i] = person i의 null_in 어텐션 (장면 사물 봄)
[TemporalGraphBlock]  per-person MHA over T frames (Graph 모드와 공유)
```

총 4개 반복 → `img_layers[0..3]`, `gaze_layers[0..3]`(trunk) 수집  
Graph 모드는 추가로 `sa_layers[0..3]`(undirected SA 분기) 수집  
Graph/Hypergraph 모드는 `alpha_null_in_list[0..3]`, `alpha_null_out_list[0..3]` 수집 (dual-null loss 전용)

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

## Step 5: Social Gaze 디코딩 (모드별 분기)

```python
# 공통: gaze_projs로 4 stage trunk token projection+concat
proj_tokens = cat([gaze_projs[i](gaze_layer) for i in range(4)])  # (B*T, N, 512)

# 공통: in-out 분류 (GWS fusion 적용 전 trunk 사용)
inout = inout_decoder(proj_tokens.view(B*T*N, -1))   # (B*T*N, 1)
```

**예측은 전 모드 통합** (`decoder_lah` / `decoder_sa` 공유, LAEO는 derive):

```python
# LAH path tokens = trunk(proj_tokens). SA path tokens:
#   graph 모드  → sa_projs로 sa_layers 투영+concat (전용 undirected 분기)
#   그 외      → trunk(proj_tokens) 재사용
sa_tokens = cat([sa_projs[i](sa_layers[i]) for i in range(4)]) if use_graph else proj_tokens

# LAH: 비대칭 [h_i ‖ h_j] (방향 i→j)
pair_lah = cat([proj_tokens[:, src], proj_tokens[:, dst]], dim=-1)  # (B*T*P, 1024)
lah   = decoder_lah(pair_lah).view(B*T, num_pairs)

# SA: 대칭 [s_i+s_j ‖ |s_i−s_j|]
pair_sym = cat([sa_tokens[:, src] + sa_tokens[:, dst],
                (sa_tokens[:, src] - sa_tokens[:, dst]).abs()], dim=-1)
coatt = decoder_sa(pair_sym).view(B*T, num_pairs)

# LAEO ⟺ mutual LAH = logit-space AND (min) of both directions (전 모드 동일)
laeo = minimum(lah, lah[:, rev_idx])   # rev_idx: (d,s) index for each (s,d), n별 캐시
```

핵심:
- **LAEO는 더 이상 전용 decoder가 아님.** `min(LAH_ij, LAH_ji)` — transformer의 검증된 공식. `decoder_laeo`는 `__init__`에만 남은 미사용 dead weight(체크포인트 호환용).
- **graph 모드만 SA를 trunk가 아닌 전용 undirected 분기 토큰으로 예측.** transformer/hypergraph는 `sa_tokens == proj_tokens`(trunk).
- **null routing(`alpha_null_in/out`)은 예측에 안 들어감.** `compute_dual_null_loss`의 auxiliary supervision 전용 (graph 모드).  
  (과거의 `lah_null_proj`·`sa_null_w`·`decoder_*_gws`·hypergraph temp scalar/`attn_layer_logits` 경로는 모두 제거됨.)

**GWS (`use_gws=true`)**: heatmap·scene을 detach해 per-person `gaze_scene` embedding 추출 →
`proj_tokens = gaze_fusion(cat([proj_tokens, gaze_scene_proj(gaze_scene)]))`로 trunk만 융합 (공유 decoder 그대로, SA 분기엔 미적용).

---

## 핵심 모듈 파일 위치

| 모듈 | 파일 | 클래스/함수 |
|------|------|------------|
| 전체 MTGS 네트워크 | `mtgs/networks/mtgs_net.py` | `MTGS` |
| Lightning 학습 모델 | `mtgs/networks/models.py` | `MTGSModel` |
| Gaze 인코더 | `mtgs/networks/mtgs_net.py` | `GazeEncoder` |
| ViT-Adaptor 블록 | `mtgs/networks/adaptor_modules.py` | `InteractionBlock` |
| Social Graph (directed, LAH/LAEO) | `mtgs/networks/adaptor_modules.py` | `SocialGraphBlock` |
| Social Graph (undirected, SA) | `mtgs/networks/adaptor_modules.py` | `UndirectedSocialGraphBlock` |
| Hypergraph Block | `mtgs/networks/adaptor_modules.py` | `HypergraphBlock` |
| Temporal Graph | `mtgs/networks/adaptor_modules.py` | `TemporalGraphBlock` |
| DPT 히트맵 디코더 | `mtgs/networks/mtgs_net.py` | `ConditionalDPTDecoder` |
| Social 디코더 (Transformer/Graph 전용) | `mtgs/networks/mtgs_net.py` | `LinearDecoderSocialGraph` |
| InOut 디코더 (전 모드 공유) | `mtgs/networks/mtgs_net.py` | `InOutDecoder` |

---

## 손실 함수 구성 (losses.py)

```python
total_loss = (
    3   * angular_loss    # 시선 방향 코사인 손실
  + 100 * dist_loss       # 시선 포인트 L2 거리 (heatmap mode에서는 0)
  + 1000 * heatmap_loss   # 히트맵 MSE
  + 2   * inout_loss      # in/out BCE
  + 1   * lah_loss        # LAH BCE (pos_weight=3)
  + 1   * laeo_loss       # LAEO BCE (min-derive → gradient는 decoder_lah로 흐름)
  + 1   * coatt_loss      # SA BCE
)
# graph 모드 추가 (null 노드 활성 시):
loss += lambda_null * (loss_null_out + loss_null_in)   # compute_dual_null_loss, lambda_null=0.5
```

UCO-LAEO, VideoCoAtt 데이터에는 heatmap/angular 손실에 0.1 가중치 적용 (gaze GT 품질이 낮기 때문).

---

## 학습 최적화 세부사항

- **옵티마이저**: AdamW (weight_decay=1e-3)
- **학습률 (transformer 모드)**: 기본 1e-6, `gaze_encoder_temporal/people_temporal/decoder_sa` 3×
- **학습률 (graph 모드, 5 param groups)**: 기본 1e-6, `gaze_encoder_temporal` 3×, `social_graph_blocks`+`sa_social_blocks` 10×, `temporal_graph_blocks`+`sa_temporal_blocks` 5×, `decoder_lah/decoder_laeo/decoder_sa/sa_projs`(+GWS시 `gaze_scene_proj/gaze_fusion`) 3×, 나머지 base. → `train_vsgaze.sh`의 SWA lr 5개와 정합.
- **학습률 (hypergraph 모드, 4 param groups)**: 기본 1e-6, `gaze_encoder_temporal/temporal_graph_blocks/decoder_lah/decoder_laeo/decoder_sa`(+GWS) 3×, `hypergraph_blocks`는 base LR.
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
