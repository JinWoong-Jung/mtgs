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
  num_samples: 36000      # 참고용 메모(GazeFollow: 108955, VSGaze: 36000) — 코드에서 실제로 읽지 않는 죽은 키
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
    gaze_encoder: False
    gaze_encoder_backbone: False
    image_tokenizer: True   # 미사용 모듈(forward에서 호출 안 됨)이라 실질 효과 없음
    vit_encoder: True
    vit_adaptor: False
    gaze_decoder: False     # output="point"일 때만 존재하는 모듈; heatmap 모드에선 미생성
    inout_decoder: False
    all_but_gaze_graph: False   # true면 위 플래그 전부 무시하고 trunk 전체 freeze + gaze_graph_block만 학습
  checkpoint_monitor: "metric/val/dist"   # train_vsgaze.sh는 "metric/val/social_ap"로 오버라이드
  checkpoint_mode: "min"                   # social_ap 사용 시 "max"
```

> `depth_tokenizer` freeze 키는 존재하지 않는다 — config에 있어도 코드에서 읽지 않는 무효 키다.
> SWA(Stochastic Weight Averaging)는 코드/콜백 어디에도 없다.

---

## 검증/테스트 설정

```yaml
val:
  batch_size: 8

test:
  checkpoint: null          # CLI에서 test.checkpoint=<path>로 오버라이드
  batch_size: 1             # num_people="all"(가변 N)이라 배치=1로 cross-sample padding 이슈 회피
```

`test.max_people` 같은 상한 필터는 없다 — variable-N 문제는 batch_size=1로 우회한다 (자세한 내용은
[architecture.md](architecture.md)의 "Test 배치 처리" 참조).

---

## Gaze Graph 설정 (social-prediction head)

```yaml
gaze_graph:
  use: true                # true: GazeGraphBlock 헤드 | false: 원본 social decoder(decoder_lah/decoder_sa)
  num_layers: 2
  edge_dim: 512             # node D(=token_dim)와 동일 차원
  detach_input: true        # social loss가 trunk(people_interaction/temporal)로 안 흐르게 firewall
  laeo_derive: "decoder"    # "decoder": head_laeo MLP | "lah_min": min(LAH_i→j, LAH_j→i)
  use_prior: true           # 4채널 기하 edge prior 사용 여부 (prior_w는 항상 zero-init)
  use_face_proj: true       # detached raw-face(GazeEncoder token) node 재주입
  use_node_geom: false      # [cx,cy,w,h,gaze_vec] geometry MLP node 재주입
  use_type_embed: true      # person/null_in/null_out target-type embedding (edge init)
  frozen: false             # true면 train.freeze.all_but_gaze_graph와 동일하게 동작
  lambda_null: 0.5          # dual-null(null_in+null_out) aux loss weight
  head_lr_mult: 100         # gaze_graph_block 전용 LR = optimizer.lr * head_lr_mult
  use_row_attn: true        # ablation: row edge-attention (source→outgoing targets)
  use_col_attn: true        # ablation: col edge-attention (target→incoming sources)
  use_temporal_attn: true   # ablation: graph 자체의 per-edge temporal attention (T>1일 때만)
  use_null_in: true         # ablation: in-frame scene-object null node
  use_null_out: true        # ablation: out-of-frame null node
```

`use_row_attn`/`use_col_attn`/`use_null_in`/`use_null_out`/`use_face_proj`/`use_node_geom`/`use_type_embed`는
capacity-controlled ablation (모듈은 항상 생성, off면 forward에서 기여만 0으로 마스킹 → 체크포인트 호환 유지).
`use_temporal_attn`만 module-skip 방식(off면 모듈 자체 미생성). `prior_weight` 키는 더 이상 없음 —
`prior_w`는 항상 zero-init 학습 파라미터. 수식/버전 이력은 [gaze_graph_math.md](gaze_graph_math.md),
[version.md](version.md) 참조.

> **2026-06-13 제거됨**: `interaction.type` (`transformer`/`graph`/`hypergraph`)과 `SocialGraphBlock`/
> `UndirectedSocialGraphBlock`/`HypergraphBlock` 기반 구조는 완전히 삭제되고 `gaze_graph`(GazeGraphBlock)
> 단일 아키텍처로 리팩토링됐다. 과거 `interaction:` config 섹션은 더 이상 존재하지 않는다 —
> 관련 문서는 [interaction_module.md](interaction_module.md)에 역사적 참고용으로만 남아있다.

---

## 옵티마이저/스케줄러

```yaml
optimizer:
  lr: 1e-6
  weight_decay: 1e-3

scheduler:
  type: CosineAnnealingLR   # "CosineAnnealingLR": step 단위 linear warmup→cosine decay | "constant": 고정 LR
  warmup_epochs: 3          # linear warmup 길이 (epoch 단위, step으로 환산되어 적용)
  eta_min: 0                # cosine decay 최종 LR 하한
```

---

## CLI 오버라이드 예시

```bash
# test 단독 실행
python main.py experiment.task=test \
    test.checkpoint=/path/to/best.ckpt

# gaze_graph ablation 예시 (row/col attention 끄기)
python main.py experiment.task=train+test \
    gaze_graph.use_row_attn=false \
    gaze_graph.use_col_attn=false
```

---

## 실험 출력 경로

Hydra가 `../experiments/{date}/{experiment.name}/` 디렉토리를 자동 생성.  
체크포인트: `{output_folder}/train/checkpoints/{best,last}.ckpt`
