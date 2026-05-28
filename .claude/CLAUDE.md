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

## 현재 체크포인트 경로

| 용도 | 경로 |
|------|------|
| **VSGaze best (현재 사용 중)** | `/home/jinwoongjung/MTGS/experiments/2026-05-18/MTGS-dinov2-vitb14-448-VSGaze/train/checkpoints/best.ckpt` |
| GazeFollow pretrain (Stage1) | `/home/jinwoongjung/MTGS/experiments/2026-05-16/MTGS-dinov2-vitb14-448/train/checkpoints/best.ckpt` |

`test_vsgaze.sh`의 `CHECKPOINT` 변수가 VSGaze best를 가리키고 있음.

---

## 모델 아키텍처 (mtgs_net.py)

자세한 내용은 [architecture.md](.claude/architecture.md) 참조.

### 핵심 구성 요소

1. **GazeEncoder** — ResNet-18 백본으로 각 사람의 head crop 인코딩 → gaze token + gaze vector
2. **Temporal Gaze Attention** — 시간적 컨텍스트에서 gaze token self-attention
3. **DINOv2 (frozen ViT-B/14)** — scene encoder
4. **ViT-Adaptor (InteractionBlock × 4)** — scene↔person cross-attention
5. **People Interaction** — person token 간 self-attention (**Transformer 모드 전용**)
6. **People Temporal** — 시간 축 person token self-attention (**Transformer 모드 전용**)
7. **SocialGraphBlock × 4** — outgoing directed graph message passing (**Graph 모드 전용**): node feature 업데이트만 수행, social 예측은 downstream decoder에 위임
8. **TemporalGraphBlock × 4** — per-person MHA over T frames (**Graph 모드 전용**)
9. **ConditionalDPTDecoder** — multi-scale 디코더 → gaze heatmap
10. **`gaze_projs` + `InOutDecoder`** — in/out 분류 헤드 (**Transformer/Graph 모드 공유**)
11. **`decoder_lah` + `decoder_sa` (LinearDecoderSocialGraph)** — LAH, SA pair-wise 분류 헤드 (**Transformer/Graph 모드 공유**; Transformer checkpoint에서 warm-start)

> **Graph 모드**: `interaction.type=graph`로 전환. SocialGraphBlock/TemporalGraphBlock이 People Interaction/Temporal을 대체.  
> Social prediction (LAH/SA/LAEO)은 양 모드 모두 동일한 pair-wise decoder 경로 사용.  
> **Prior weight**: `SocialGraphBlock`의 `prior_w_attn` 1개만 learnable (init=0.5). Attention routing에만 적용.

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
sbatch scripts/train_gazefollow.sh

# Stage 2: VSGaze 파인튜닝 (train+test 자동 실행)
sbatch scripts/train_vsgaze.sh   # WEIGHTS에 Stage1 ckpt 경로 설정

# test 단독 실행
sbatch scripts/test_vsgaze.sh    # CHECKPOINT에 대상 ckpt 경로 설정
```

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
test:
  batch_size: 4
  max_people: 11   # null=제한 없음, 정수=초과 샘플 스킵
```

---

## 환경

- conda 환경: `mtgs` (Python 3.10)
- SLURM GPU: `rtx6000` (97GB VRAM)
- W&B project: `gaze-social/MTGS`
