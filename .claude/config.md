# 설정 파일 가이드 (config.yaml)

설정 파일: `mtgs/config/config.yaml`  
설정 관리: Hydra + OmegaConf

---

## 사용 전 필수 수정 항목

| 항목 | 설명 |
|------|------|
| `wandb.username` | W&B 사용자명 |
| `model.gaze_weights` | ResNet-18 Gaze360 사전학습 가중치 경로 |
| `model.weights` | 모델 초기화 체크포인트 (False면 scratch) |
| `data.ann_root` | HDF5 어노테이션 파일 디렉토리 |
| `data.*.root` | 각 데이터셋 이미지 루트 경로 |

---

## 모델 설정

```yaml
model:
  patch_size: 14          # DINOv2 ViT-B/14 패치 크기
  head_size: 224          # head crop 입력 크기 (GazeEncoder)
  token_dim: 768          # ViT 토큰 차원
  image_size: 448         # 장면 이미지 입력 크기
  gaze_feature_dim: 512   # GazeEncoder 특징 차원
  encoder_depth: 12       # ViT 블록 수
  decoder_hooks: [2, 5, 8, 11]
  decoder_hidden_dims: [96, 192, 384, 768]
```

---

## 데이터 설정

```yaml
data:
  num_people: 4           # train/val 배치당 최대 사람 수 (test는 "all")
  image_size: 448
  temporal_stride: 3
  temporal_context: 2     # window_size = 2*2+1 = 5 프레임
  heatmap_size: 64
  num_samples: 36000      # GazeFollow: 108955, VSGaze: 36000
```

---

## 실험 설정

```yaml
experiment:
  task: train+test   # "train", "test", "val", "train+test" 조합 가능
  dataset: vsgaze    # gazefollow, vat, childplay, videocoatt, uco_laeo, vsgaze
```

---

## 학습 설정

```yaml
train:
  seed: 101
  precision: bf16-mixed
  epochs: 20
  batch_size: 8
  freeze:
    image_tokenizer: True   # DINOv2 관련 frozen (기본값)
    vit_encoder: True
  swa:
    use: True
    epoch_start: 12
    annealing_epochs: 6
```

---

## 검증/테스트 설정

```yaml
val:
  batch_size: 8

test:
  checkpoint: null          # CLI에서 test.checkpoint=<path>로 오버라이드
  batch_size: 4             # test는 가변 N 패딩 때문에 train보다 작게 설정
  max_people: 11            # 이 수보다 많은 사람이 있는 샘플은 스킵
                            # null = 제한 없음 (VRAM 변동성 심함 주의)
                            # VSGaze test 기준: N≤11 → 95.3% 샘플 커버
```

**`max_people` 동작 흐름:**
`config.yaml` → `dataset.py:build_dataset()` → `VSGazeDataModule(max_people=11)`  
→ `test_dataloader()` → `_filter_by_max_people()` → `Subset`으로 필터링

---

## Interaction 설정

```yaml
interaction:
  type: hypergraph      # "transformer" | "graph" | "hypergraph"
  order: "extract_first" # "inject_first" (원본 순서) | "extract_first" (신규)
  num_layers: 2         # HypergraphBlock / SocialGraphBlock 내부 iteration 수

  graph:                # graph 모드 전용
    aggr: "outgoing"
    use_null_node: true   # dual-null (loss 전용; 예측엔 미사용)
    lambda_null: 0.5
    use_gaze_prior: true  # directed 블록 LAH 방향 prior
    prior_weight: 0.5
    use_sa_prior: true    # ★ 이제 undirected SA 블록의 gaze·gaze prior를 제어 (directed 블록은 use_sa_prior=False 고정)
    sa_prior_weight: 0.5
  use_gws: false          # GWS: trunk에 gaze_scene_proj+gaze_fusion 융합 (별도 _gws decoder 없음)
```

**Graph 모드 구조 (2-graph 분리)**: directed `SocialGraphBlock`(LAH/LAEO/heatmap) + 전용 undirected `UndirectedSocialGraphBlock`(SA). LAEO는 `min(LAH_ij, LAH_ji)`로 derive (`decoder_laeo` 미사용). 자세한 내용은 [interaction_module.md](.claude/interaction_module.md), [architecture.md](.claude/architecture.md) 참조.

**`interaction.order` 동작:**
- `inject_first`: Injector → ViT → Extractor → Social (원본)
- `extract_first`: Extractor → Social → Injector → ViT (scene-aware social interaction)

**모드 간 호환 주의**: GazeFollow Stage1과 VSGaze Stage2는 반드시 동일한 `type` + `order` 조합 사용.

---

## 옵티마이저/스케줄러

```yaml
optimizer:
  lr: 1e-6
  weight_decay: 1e-3

scheduler:
  type: CosineAnnealingWarmRestarts
  warmup_epochs: 4
  t_0_epochs: 20
```

---

## CLI 오버라이드 예시

```bash
# test 단독 실행
python main.py experiment.task=test \
    test.checkpoint=/path/to/best.ckpt \
    test.max_people=11

# batch_size, max_people 조정
python main.py experiment.task=test \
    test.checkpoint=... \
    test.batch_size=8 \
    test.max_people=null
```

---

## 실험 출력 경로

Hydra가 `../experiments/{date}/{experiment.name}/` 디렉토리를 자동 생성.  
체크포인트: `{output_folder}/train/checkpoints/{best,last}.ckpt`
