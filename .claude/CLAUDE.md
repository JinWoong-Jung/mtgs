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
│   │   └── vsgaze.py
│   ├── networks/
│   │   ├── mtgs_net.py       # MTGS 모델 아키텍처 (핵심)
│   │   ├── models.py         # PyTorch Lightning MTGSModel (학습/평가 루프)
│   │   │                     # 모든 social 메트릭 .cpu() 누적으로 변경
│   │   └── adaptor_modules.py
│   ├── train/
│   │   ├── dataset.py        # build_dataset()
│   │   ├── collate.py        # pad_collate_fn (VLM 파이프라인 전용)
│   │   ├── losses.py
│   │   ├── trainer.py
│   │   ├── callbacks.py
│   │   └── transforms.py
│   └── utils/
├── scripts/
│   ├── main.py
│   ├── main_llm.py
│   ├── train_vsgaze.sh       # VSGaze 파인튜닝 (FROZEN=true/false로 post-training 겸용)
│   ├── train_llm_align.sh
│   └── test_vsgaze.sh
└── logs/
```

> 2026-07: `train_gazefollow.sh`, `train_postgraph.sh`, `test_gazefollow.sh`, `test_vat.sh` 삭제.
> `train_postgraph.sh`는 `train_vsgaze.sh`에서 `FROZEN=true` + `gaze_graph.frozen=true` 조합으로 대체 가능해 중복 제거. GazeFollow stage-1 재학습이 필요하면 `train_vsgaze.sh`를 `experiment.dataset=gazefollow`로 오버라이드해서 사용.

---

## 현재 체크포인트

| 용도 | 경로 |
|------|------|
| GazeFollow stage-1 | `experiments/2026-06-05/GazeFollow_gaze_graph/train/checkpoints/best.ckpt` |
| VSGaze stage-2 warm-start | `weights/mtgs-vsgaze.ckpt` (구 transformer 기반; gaze_graph 재학습 예정) |

> 2026-06-13: transformer/graph/hypergraph 모드 완전 제거, gaze_graph 단일 아키텍처로 리팩토링 완료.
> 기존 transformer/graph/hypergraph 체크포인트는 더 이상 로드 불가 (의도된 것).

---

## 모델 아키텍처 (mtgs_net.py)

자세한 내용은 [architecture.md](.claude/architecture.md) 참조.

### 핵심 구성 요소

1. **GazeEncoder** — ResNet-18 백본으로 각 사람의 head crop 인코딩 → gaze token + gaze vector
2. **Temporal Gaze Attention** — 시간적 컨텍스트에서 gaze token self-attention
3. **DINOv2 (frozen ViT-B/14)** — scene encoder
4. **ViT-Adaptor (InteractionBlock × 4)** — scene↔person cross-attention. inject_first 고정 순서: Injector → ViT → Extractor → Social
5. **People Interaction** — person token 간 self-attention (ViT-Adaptor 각 단계마다 실행)
6. **People Temporal** — 시간 축 person token self-attention (ViT-Adaptor 각 단계마다 실행)
7. **ConditionalDPTDecoder** — multi-scale 디코더 → gaze heatmap (64×64). `img_layers` + `gaze_layers`(trunk) 입력.
8. **`gaze_projs` + `InOutDecoder`** — in/out 분류 헤드
9. **GazeGraphBlock × 1** — ViT-Adaptor 4단계 완료 후 concat된 proj_tokens(D×4=512)에 적용. Edge tensor E ∈ ℝ^{B×T×N×(N+2)×D_e}를 row/col attention으로 반복 정제 → LAH/LAEO/SA/null을 edge readout head로 직접 출력. 수식: [gaze_graph_math.md](.claude/gaze_graph_math.md).  
    - **LAH**: `head_lah(E[i→j])` (directed edge 단독)  
    - **LAEO**: `head_laeo([E[i→j]; E[j→i]])` symmetrized — 전용 MLP  
    - **SA**: `head_sa([ni_i ‖ ni_j ‖ |ni_i−ni_j| ‖ E[i→j] ‖ E[j→i]])` symmetrized — ni = E[·→null_in] (장면 응시 패턴)  
    - **null_in/out**: `head_null_in/out(E[i→null])` (explicit null edge heads)

> **Social Prediction 경로:**
>
> | 태스크 | 방법 |
> |--------|------|
> | LAH | `head_lah(E[i→j])` — directed edge 단독 |
> | LAEO | `head_laeo([E[i→j]; E[j→i]])` symmetrized — 전용 MLP |
> | SA | `head_sa([ni_i ‖ ni_j ‖ |diff| ‖ E[i→j] ‖ E[j→i]])` symmetrized (ni=null_in edge) |
> | null | `head_null_in/out(E[i→null])` 직접 예측 → dual-null aux loss |
>
> `GazeGraphBlock.forward()`가 lah/laeo/sa/null_in/null_out 모두 직접 반환.

> **`gaze_graph.use=false` (원본 social decoder):** GazeGraphBlock 대신 `decoder_lah`/`decoder_sa`가
> person token pair에서 직접 예측. LAH=`decoder_lah([h_i‖h_j])`, SA=`decoder_sa([s_i+s_j‖|s_i−s_j|])`,
> LAEO=`min(LAH_ij, LAH_ji)`. null edge 없음(null_in/out=None → dual-null loss 자동 skip).
> 옵티마이저 param group은 `gaze-graph-block` 대신 `social-decoder`로 교체(4 group 구조 동일).

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

---

## 학습 파이프라인

```bash
# Stage 1: GazeFollow 사전학습 (experiment.dataset=gazefollow로 오버라이드)
sbatch scripts/train_vsgaze.sh       # EXP_NAME, WEIGHTS 상단에서 설정

# Stage 2: VSGaze 파인튜닝 (train+test 자동 실행)
sbatch scripts/train_vsgaze.sh       # WEIGHTS, EXP_NAME, LAEO_DERIVE 설정

# Post-training: trunk frozen, gaze_graph_block만 재학습
# → train_vsgaze.sh에서 FROZEN=true (+ WEIGHTS를 VSGaze ckpt로) 설정하고 그대로 실행
sbatch scripts/train_vsgaze.sh

# test 단독 실행
sbatch scripts/test_vsgaze.sh        # CHECKPOINT에 대상 ckpt 경로 설정
```

---

## Test 시 가변 인원 처리

train은 `num_people=4` 고정, test는 `num_people="all"`(가변). test `batch_size=1`이므로 cross-sample collation 문제는 없음.

**적용된 사항:**
- **`models.py` 메트릭 `.cpu()` 누적** — AUROC, AveragePrecision 6개 모두 CPU 텐서로 업데이트 → GPU 메모리 점진적 증가 방지.

---

## 설정 (config.yaml)

주요 설정:

```yaml
gaze_graph:
  use: true                 # true: GazeGraphBlock 헤드 | false: 원본 social decoder (decoder_lah/decoder_sa)
  num_layers: 2
  edge_dim: 512              # node D와 동일 차원 (V14~)
  detach_input: true         # social loss가 trunk(people_interaction/temporal)로 안 흐르게 firewall
  laeo_derive: "decoder"     # "decoder": head_laeo MLP | "lah_min": min(LAH_i→j, LAH_j→i)
  use_prior: true            # 기하 edge prior(4채널) 사용 여부; prior_w는 zero-init 학습 스칼라
  use_face_proj: true        # detached raw-face(GazeEncoder token) node 재주입
  use_node_geom: false       # [cx,cy,w,h,gaze_vec] geometry MLP node 재주입
  use_type_embed: true       # person/null_in/null_out target-type embedding (edge init)
  frozen: false
  lambda_null: 0.5           # dual-null(null_in+null_out) aux loss weight
  head_lr_mult: 100          # gaze_graph_block 전용 LR 배수 (graph_lr = optimizer.lr * head_lr_mult)
  use_row_attn: true         # ablation: source i가 자신의 outgoing edges attend
  use_col_attn: true         # ablation: target k가 incoming edges attend
  use_temporal_attn: true    # ablation: graph 자체의 per-edge temporal attention (step ⑥, T>1일 때만)
  use_null_in: true          # ablation: in-frame scene-object null node
  use_null_out: true         # ablation: out-of-frame null node
```

`prior_weight`는 제거됨 — `prior_w`는 항상 zero-init 학습 파라미터(`face_proj`/`node_geom_mlp`와 동일한 "safe no-op" 초기화 패턴)이므로 config로 초기값을 줄 필요가 없다.
`use_row_attn`/`use_col_attn`/`use_null_in`/`use_null_out`/`use_face_proj`/`use_node_geom`/`use_type_embed`는 모두 **capacity-controlled ablation**: 모듈/파라미터는 항상 생성되고, 비활성 시 forward에서 해당 항목의 기여만 0으로 마스킹한다 (체크포인트 호환성 유지). 유일하게 `use_temporal_attn=false`는 모듈 자체를 생성하지 않는 module-skip 방식.

```yaml
vlm:                        # LLM alignment stage (별도 파이프라인)
  ...

test:
  batch_size: 1
```

**Post-training (trunk frozen) — `train_vsgaze.sh`의 `FROZEN` 변수:**
```bash
FROZEN=true    # true: trunk FREEZE + gaze_graph_block만 학습 (stage-2 VSGaze ckpt 권장)
               # false: 전체 joint training (stage-1 GazeFollow ckpt 권장)
```

---

## 환경

- conda 환경: `mtgs` (Python 3.10)
- SLURM GPU: `rtx6000` (97GB VRAM)
- W&B project: `gaze-social/MTGS`
