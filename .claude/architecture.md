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

**GazeGraphBlock 내부 흐름 (V14+, 현재 코드 기준):**

```
[Step 1] 통합 node init (src/tgt 공유, heatmap XAttn 없음):
           face  = use_face_proj ? face_proj(gaze_feat.detach()) : 0   ← raw GazeEncoder token 재주입
           geom  = use_node_geom ? node_geom_mlp([cx,cy,w,h,gaze_vec]) : 0
           node  = LN(person_token + face) + geom
           v_tgt = node_proj(cat[node, null_in_node, null_out_node])   # (Tl=N+2, De)
           v_src = v_tgt[:N]   ← persons as sources (src/tgt 동일 표현 공유)

[Step 2] Edge init (4채널 기하 prior + type embedding):
           prior = [cos(g_i,dir_ij), heatmap_overlap(i→j), rel_pos_dx, rel_pos_dy]  ← person 타깃만
                   null_in : [1-Σoverlap, 0, 0, 0]     (heatmap이 사람에 안 걸리는 비율)
                   null_out: [1-sigmoid(inout_logit), 0, 0, 0]
           edge_feat = use_prior ? prior_w(zero-init) · linear_edge(prior) : 0
           type_exp  = use_type_embed ? type_emb[person|null_in|null_out] : 0
           E_init = MLP([v_src ‖ v_tgt ‖ edge_feat ‖ type_exp]) × ev

[Step 3] _UnifiedRefiner × L(=2) layers, 매 레이어 독립 가중치, 총 6단계:
           ① Row-attn   (use_row_attn: source i가 자신의 Tl outgoing edges attend)
           ② Col-attn   (use_col_attn: target k가 N incoming sources attend; null_out 제외)
           ③ Edge refresh: LN(E_in + MLP([row_ctx ‖ col_ctx]))
           ④ Node update (attention pooling, row/col 방향별 게이팅):
               v_src_i ← out_agg 기반 (use_row_attn),  v_tgt_j / v_null_in ← in_agg 기반 (use_col_attn)
           ⑤ Node re-injection: LN(E + MLP([E ‖ v_src ‖ v_tgt])) — row/col 둘 다 off면 skip
           ⑥ Temporal edge attn (use_temporal_attn, T>1일 때만): 각 edge가 자신의 T프레임을 attend
             — MTGS의 people_temporal과는 별개, 그래프 자체의 edge-level temporal consistency

           row/col/null_in/null_out/face_proj/node_geom/type_embed 는 모두 capacity-controlled
           ablation: 모듈·파라미터는 항상 생성되고, off일 때 forward에서 기여만 0으로 마스킹된다
           (체크포인트가 4개 ablation 설정 어디서든 그대로 로드됨). use_temporal_attn만 예외로,
           off면 모듈 자체를 생성하지 않는 module-skip.

[Step 4] Readout (E_pp = E[:,:,:,:N,:] person-to-person 부분):
           LAH      : head_lah(E[i→j])                           — directed edge 단독
           LAEO     : head_laeo([E[i→j]; E[j→i]]) symmetrized    — 전용 MLP (laeo_derive="decoder"일 때만;
                        "lah_min"이면 head_laeo forward 자체를 skip하고 min(LAH_ij,LAH_ji)로 mtgs_net.py에서 derive)
           SA       : head_sa([ni_i ‖ ni_j ‖ |diff| ‖ E[i→j] ‖ E[j→i]]) symmetrized
                        ni = E[·→null_in] (장면 응시 패턴), 5·De
           null_in  : head_null_in(E[i→null_in])
           null_out : head_null_out(E[i→null_out])
```

과거(≤V13) 설계는 소스 노드에 heatmap cross-attention, 타깃 노드에 overlap-가중 message pre-init을 따로 두었으나 V14에서 person_token+face+geom 단일 통합 init으로 단순화됐다. 버전별 변경 이력은 [version.md](version.md), 수식 상세는 [gaze_graph_math.md](gaze_graph_math.md) 참조.

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
- **학습률 (4 param groups, `models.py:configure_optimizers`)**:
  - `gaze-encoder-temporal` — base × 3
  - `people-temporal` — base × 3
  - `gaze-graph-block` (또는 `use=false`일 때 `social-decoder`) — base × `gaze_graph.head_lr_mult` (기본 100; random-init + `detach_input=true`라 trunk보다 훨씬 큰 LR 필요)
  - `base` (나머지, 기본 1e-6)
- **스케줄러**: `scheduler.type`으로 선택
  - `CosineAnnealingLR`(기본): step 단위 linear warmup(`warmup_epochs`, 기본 3) → cosine decay(`eta_min`, 기본 0). 모든 param group이 각자의 초기 LR 기준으로 동일 비율로 스케일됨
  - `constant`: warmup/decay 없이 각 group이 생성 시 LR을 그대로 유지
- **SWA**: 사용 안 함 (코드/콜백에 없음)
- **정밀도**: bf16-mixed
- **Gradient Clipping**: 사용 안 함 (`gradient_clip_val: None`, `trainer.py`에 하드코딩 — config로 켜지 않는다)
- **Frozen 레이어 (기본값)**: `image_tokenizer`, `vit_encoder` — 단, `image_tokenizer`는 forward에서 아예 호출되지 않는 미사용 모듈이라 freeze 여부가 실질적 의미 없음 (scene encoder는 DINOv2)
- **`gaze_graph.frozen=true` 또는 `train.freeze.all_but_gaze_graph=true` 시**: trunk 전체 freeze(BN도 eval 고정) + `gaze_graph_block`(또는 `use=false`면 `decoder_lah`/`decoder_sa`) 파라미터만 학습

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

## Test 배치 처리

test 시 `num_people="all"`로 샘플마다 N이 다르지만, `test.batch_size=1`(config 기본값)이라 배치 내
cross-sample padding이 필요 없어 PyTorch 기본 collate로 충분하다. `DataLoader`에 커스텀
`collate_fn`은 전달하지 않는다 (`mtgs/datasets/vsgaze.py:test_dataloader`).

> `mtgs/train/collate.py::pad_collate_fn`은 정의만 되어 있고 저장소 어디에서도 import되지 않는
> 미사용 함수다. 가변-N 배치가 필요해지면(`test.batch_size>1`) 이 함수를 다시 연결해야 한다.
