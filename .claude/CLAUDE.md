# MTGS: Multi-Person Temporal Gaze Following and Social Gaze Prediction

**논문**: NeurIPS 2024 (Idiap Research Institute)  
**저자**: Anshul Gupta, Samy Tafasca, Arya Farkhondeh, Pierre Vuillecard, Jean-Marc Odobez

---

## 개요

MTGS는 비디오 입력에서 **여러 사람 각각의 시선(gaze)** 을 동시에 예측하고, 사람들 간의 **사회적 시선(social gaze)** 관계를 추론하는 프레임워크다.

**주요 출력값:**
- `gaze_heatmap` — 각 사람의 시선이 향하는 위치를 나타내는 64×64 히트맵
- `gaze_vec` — 시선 방향 단위 벡터 (x, y)
- `inout` — 시선이 화면 내부를 향하는지 여부 (in/out 분류)
- `lah` (Looking at Head) — 어떤 사람이 다른 사람의 머리를 보고 있는지 (pair-wise)
- `laeo` (Looking at Each Other) — 두 사람이 서로를 보고 있는지 (pair-wise)
- `coatt` / SA (Shared Attention) — 두 사람이 같은 곳을 보고 있는지 (pair-wise)

---

## 프로젝트 구조

```
MTGS/
├── mtgs/
│   ├── config/               # Hydra 설정 (config.yaml)
│   ├── datasets/             # 각 데이터셋 DataModule
│   │   └── vsgaze.py         # ★ _filter_by_max_people(), pad_collate_fn 연결
│   ├── networks/
│   │   ├── mtgs_net.py       # MTGS 모델 아키텍처 (핵심)
│   │   ├── models.py         # PyTorch Lightning MTGSModel (학습/평가 루프)
│   │   │                     # ★ 모든 social 메트릭 .cpu() 누적으로 변경
│   │   └── adaptor_modules.py
│   ├── train/
│   │   ├── dataset.py        # build_dataset() — ★ max_people 파라미터 추가
│   │   ├── collate.py        # ★ 신규: pad_collate_fn (가변 N 배치 패딩)
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── transforms.py
│   └── utils/
├── scripts/
│   ├── main.py
│   ├── train_gazefollow.sh
│   ├── train_vsgaze.sh       # VSGaze 학습+테스트 (train+test)
│   └── test_vsgaze.sh        # ★ 신규: test 단독 실행 스크립트
└── logs/
```

---

## 현재 체크포인트 / 실험 경로

| 용도 | 경로 |
|------|------|
| VSGaze best (transformer, baseline) | `experiments/(baseline)VSGaze_transformer/` |
| graph (prior on, gws off) | `experiments/graph/VSGaze_graph_inject_first/` |
| graph (prior off, gws on) | `experiments/graph_no_prior/VSGaze_graph_inject_first/` |
| hypergraph | `experiments/hypergraph/VSGaze_hypergraph_inject_first/` |

각 디렉토리의 `metric_calculation_*.out` 마지막 부분에 LAH/LAEO/SA AP·AUC·F1 정리됨.

### VSGaze test 실측 (AP 기준, 2026-06-03 시점)

| | transformer | graph(prior) | graph(no_prior) | hypergraph |
|---|---|---|---|---|
| Dist↓ | **0.0881** | 0.0908 | 0.0887 | 0.0887 |
| LAEO AP | **0.8024** | 0.7354 | 0.7600 | 0.7750 |
| LAH AP | **0.8982** | 0.8913 | 0.8886 | 0.8274 |
| SA AP | **0.6114** | 0.6104 | 0.5812 | 0.3705 |

→ 직전 graph 변종은 social 모든 AP에서 transformer **미달**(특히 LAEO 붕괴, SA 동률). 이를 극복하려고 아래 코드 변경 적용.

**현재 코드 상태 (2026-06-03 변경, 미학습 — 재학습 필요):**
1. **LAEO AND-derive**: 전 모드 `laeo = min(LAH_ij, LAH_ji)`로 통일 (전용 `decoder_laeo` 미사용). transformer의 검증된 공식 회복 목적.
2. **2-graph 분리 (graph 모드)**: directed trunk(`SocialGraphBlock`, LAH/LAEO/heatmap) + 전용 undirected SA 분기(`UndirectedSocialGraphBlock`, sigmoid gated-mean). SA만 분리 표현, trunk는 보존.
3. directed 블록 `use_sa_prior=False` (SA prior는 undirected 블록으로 이동).

> 다음 런: `train_vsgaze.sh`로 graph stage2 재학습. 깨끗한 비교 위해 직전 best와 동일 config(prior on, gws off) 권장. GazeFollow stage1 ckpt warm-start 시 `sa_*` 모듈만 random init (`strict=False`).

---

## 모델 아키텍처 (mtgs_net.py)

자세한 내용은 [architecture.md](.claude/architecture.md) 참조.

### 핵심 구성 요소

1. **GazeEncoder** — ResNet-18 백본으로 각 사람의 head crop 인코딩 → gaze token + gaze vector
2. **Temporal Gaze Attention** — 시간적 컨텍스트에서 gaze token self-attention
3. **DINOv2 (frozen ViT-B/14)** — scene encoder
4. **ViT-Adaptor (InteractionBlock × 4)** — scene↔person cross-attention. `interaction.order`에 따라 두 가지 실행 순서:
   - `inject_first` (기본/원본): Injector → ViT → Extractor → Social
   - `extract_first`: Extractor → Social → Injector → ViT
5. **People Interaction** — person token 간 self-attention (**Transformer 모드 전용**)
6. **People Temporal** — 시간 축 person token self-attention (**Transformer 모드 전용**)
7. **SocialGraphBlock × 4** — outgoing directed graph (softmax+dual-null) message passing. **Graph 모드 trunk** (LAH/LAEO/heatmap/inout 담당). LAH 방향 prior만 사용(`use_sa_prior=False`).
8. **UndirectedSocialGraphBlock × 4** — 무방향 SA 그래프. 대칭 edge + **독립 sigmoid 게이트 + 이웃수 정규화 gated-mean** 집계. **Graph 모드 전용, SA(CoAtt) 담당**. trunk로 되먹이지 않는 평행 read-out 분기 (입력은 directed 블록 직전 trunk와 동일). SA gaze prior(`gaze_i·gaze_j`)는 여기 소속. `sa_temporal_blocks`(TemporalGraphBlock) + `sa_projs`(768→128) 동반.
9. **HypergraphBlock × 4** — person hyperedge(N개) + null_in/null_out 기반 메시지 패싱 (**Hypergraph 모드 전용**).
10. **TemporalGraphBlock × 4** — per-person MHA over T frames (**Graph/Hypergraph 모드 공유**)
11. **ConditionalDPTDecoder** — multi-scale 디코더 → gaze heatmap (64×64). `img_layers` + `gaze_layers`(trunk) 입력.
12. **`gaze_projs` + `InOutDecoder`** — in/out 분류 헤드 (전 모드 공유)
13. **`decoder_lah` + `decoder_sa` (LinearDecoderSocialGraph)** — LAH, SA pair-wise 분류 헤드 (전 모드 공유). **LAEO는 별도 decoder 없이 `min(LAH_ij, LAH_ji)`로 derive** (전 모드 동일). `decoder_laeo`는 `__init__`에만 남은 **미사용 dead weight**(체크포인트 호환용). 과거의 `lah_null_proj`/`sa_null_w`/`decoder_*_gws`/hypergraph temp scalar 경로는 **모두 제거됨**.
14. **GWS (Gaze-Weighted Scene)** — `interaction.use_gws=true`일 때 활성화. heatmap(detach)을 patch grid로 downsample → scene token(detach) weighted sum → per-person `gaze_scene`. `gaze_scene_proj`(D_vit→D_person) + `gaze_fusion`(2·D_person→D_person)로 **trunk(proj_tokens)에만** 융합 (공유 decoder 그대로, SA 분기엔 미적용).

> **모드별 Social Prediction 경로 (현재 코드):**
>
> | 태스크 | Transformer | Graph | Hypergraph |
> |--------|-------------|-------|------------|
> | LAH | `decoder_lah([h_i‖h_j])` (trunk) | `decoder_lah([h_i‖h_j])` (directed trunk) | `decoder_lah([h_i‖h_j])` (trunk) |
> | LAEO | `min(LAH_ij, LAH_ji)` | `min(LAH_ij, LAH_ji)` | `min(LAH_ij, LAH_ji)` |
> | SA | `decoder_sa([s_i+s_j ‖ |s_i−s_j|])` (trunk) | `decoder_sa(...)` (**undirected 분기 토큰**) | `decoder_sa(...)` (trunk) |
>
> LAH는 비대칭 concat, SA는 대칭 feature. **graph 모드만 SA 입력이 trunk가 아닌 전용 undirected 분기**(`sa_projs(sa_layers)`).  
> `alpha_null_in/out`은 **예측에 미사용** — graph 모드 `compute_dual_null_loss`의 auxiliary supervision 전용.  
> `use_gws=true` → trunk 토큰만 `gaze_fusion`으로 확장되고 **동일한 공유 decoder** 사용 (별도 `_gws` decoder 없음).

---

## 데이터셋

| 설정값 | 데이터셋 | 특이사항 |
|--------|----------|----------|
| `gazefollow` | GazeFollow | 정적 이미지, 단일 인물 위주 |
| `vat` | VideoAttentionTarget | 비디오, 시선 + in/out |
| `childplay` | ChildPlay | 비디오, 사회적 시선 포함 |
| `videocoatt` | VideoCoAtt | 비디오, shared attention |
| `uco_laeo` | UCO-LAEO | 비디오, LAEO 레이블 |
| `vsgaze` | VSGaze (복합) | vat + childplay + videocoatt + uco_laeo 혼합 |

어노테이션: HDF5 형식, `pandas.read_hdf(path, "data")`로 로드.

### VSGaze test 사람 수 분포
- 전체 43,581 샘플, 평균 N=5.2, 최대 N=39
- N≤8: 91.1%, N≤11: 95.3% (현재 max_people=11로 설정)

---

## 학습 파이프라인

```bash
# Stage 1: GazeFollow 사전학습
sbatch scripts/train_gazefollow.sh   # INTERACTION_TYPE, INTERACTION_ORDER 상단에서 설정

# Stage 2: VSGaze 파인튜닝 (train+test 자동 실행)
sbatch scripts/train_vsgaze.sh       # INTERACTION_TYPE, INTERACTION_ORDER, WEIGHTS 설정

# test 단독 실행
sbatch scripts/test_vsgaze.sh        # CHECKPOINT에 대상 ckpt 경로 설정
```

**스크립트 핵심 변수 (양 스크립트 공통):**
```bash
INTERACTION_TYPE="hypergraph"    # "transformer" | "graph" | "hypergraph"
INTERACTION_ORDER="extract_first" # "inject_first" (원본) | "extract_first"
EXP_NAME="GazeFollow_${INTERACTION_TYPE}_${INTERACTION_ORDER}"  # 자동 생성
```

GazeFollow Stage1과 VSGaze Stage2는 **반드시 동일한 INTERACTION_TYPE + INTERACTION_ORDER**로 맞춰야 함 (weight 구조 불일치 방지).

---

## Test 시 가변 인원 처리 (커스텀 구현)

train은 `num_people=4` 고정, test는 `num_people="all"`(가변) — 이로 인해 batch collation 문제 발생.

**현재 구현된 해결책:**

1. **`mtgs/train/collate.py`의 `pad_collate_fn`**  
   batch 내 max_N으로 N-dim 텐서 zero-padding, pair label은 -1로 padding (ignore_index).

2. **`vsgaze.py`의 `_filter_by_max_people()`**  
   `test.max_people` 이상인 샘플은 DataLoader 구성 전에 스킵.  
   현재 설정: `max_people=11` (전체의 ~4.7% 스킵, 95.3% 커버).

3. **`models.py` 메트릭 `.cpu()` 누적**  
   AUROC, AveragePrecision 6개 모두 CPU에 누적 → GPU 메모리 점진적 증가 방지.

---

## 설정 (config.yaml)

주요 설정 — 자세한 내용은 [config.md](.claude/config.md) 참조.

```yaml
interaction:
  type: graph            # "transformer" | "graph" | "hypergraph"
  order: "inject_first"  # "inject_first" | "extract_first"
  num_layers: 2
  use_gws: false         # Gaze-Weighted Scene token (true/false). 스크립트에서 interaction.use_gws=true 오버라이드

test:
  batch_size: 4
  max_people: 11   # null=제한 없음, 정수=초과 샘플 스킵
```

**GWS 활성화 시 추가 파라미터:** `gaze_scene_proj`, `gaze_fusion`만 추가 (별도 `_gws` decoder 없음 — 공유 decoder 사용). GWS off 체크포인트를 GWS on 모델에 warm-start 가능 (이 두 모듈만 랜덤 초기화).

---

## 환경

- conda 환경: `mtgs` (Python 3.10)
- SLURM GPU: `rtx6000` (97GB VRAM)
- W&B project: `gaze-social/MTGS`
