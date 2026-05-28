# MTGS 원본 vs 구현 코드 차이 분석

원본: `/home/jinwoongjung/MTGS_origin`  
구현: `/home/jinwoongjung/MTGS`

변경된 파일:
- `scripts/train_gazefollow.sh`, `scripts/train_vsgaze.sh`
- `mtgs/networks/mtgs_net.py`, `mtgs/networks/models.py`, `mtgs/networks/adaptor_modules.py`
- `mtgs/train/losses.py`, `mtgs/train/trainer.py`
- `mtgs/train/dataset.py` (max_people 파라미터 추가)
- `mtgs/datasets/vsgaze.py` (max_people 필터 구현)
- `mtgs/datasets/childplay_temporal.py`
- `mtgs/experiments.py`, `mtgs/config/config.yaml`
- `mtgs/performance/compute_metrics.py`

---

## [위험도: 높음] 학습 결과에 직접 영향을 주는 차이

---

### 1. `scripts/train_vsgaze.sh` — 그라디언트 클리핑 무조건 적용 ★ 핵심 원인

**원본:**
```bash
python -m mtgs.scripts.main \
    experiment.dataset=vsgaze \
    model.weights=$WEIGHTS
```
(그라디언트 클리핑 없음)

**구현 코드:**
```bash
INTERACTION_TYPE="graph"   # 또는 "transformer"

# graph/transformer 모드 구분 없이 무조건 추가됨:
    train.gradient_clip_val=1.0 \
    train.gradient_clip_algorithm=norm \
```

**영향:**
- `INTERACTION_TYPE`가 `"graph"`이든 `"transformer"`이든, 클리핑이 항상 적용됨
- 원본 transformer 모드에는 그라디언트 클리핑이 없었음
- GazeFollow 학습 (`train_gazefollow.sh`)에는 클리핑이 없어서 원본과 유사한 결과가 나왔으나, VSGaze부터 다른 경향을 보이는 이유가 **바로 이것**
- CLAUDE.md 피드백에도 기록됨: "transformer mode에서 gradient_clip_val을 추가하면 성능이 깨짐"

**GazeFollow vs VSGaze 결과 차이 설명:** GazeFollow는 클리핑 없이 학습 → 유사한 결과. VSGaze는 클리핑 적용 → 학습 경향 달라짐.

**수정 방법:** `train_vsgaze.sh`에서 transformer mode 실행 시 `gradient_clip_val` 라인 제거 또는 조건부로 적용:
```bash
if [ "$INTERACTION_TYPE" = "graph" ]; then
    GRAD_CLIP="train.gradient_clip_val=1.0 train.gradient_clip_algorithm=norm"
else
    GRAD_CLIP=""
fi
```

---

### 2. `mtgs/train/losses.py` — `social_loss()` 구현 변경

**원본 구현:**
```python
def social_loss(social_pred, social_gt, mask, pos_weight=2.0):
    social_gt = social_gt * mask          # (1) masked 위치를 0으로 강제
    num_instances = mask.sum()
    loss = F.binary_cross_entropy_with_logits(
        social_pred, social_gt,           # (2) 전체 텐서로 BCE 계산
        pos_weight=..., reduction="none"
    )
    loss = torch.mul(loss, mask).sum() / (num_instances + 1e-6)  # (3) 마스크 후 평균
    return loss
```

**구현 코드:**
```python
def social_loss(social_pred, social_gt, mask, pos_weight=2.0):
    finite_mask = mask & torch.isfinite(social_pred)   # (1) isfinite 조건 추가
    num_instances = finite_mask.sum()
    if num_instances == 0:
        return torch.tensor(0.0, ...)
    valid_pred = social_pred[finite_mask]              # (2) valid 요소만 추출
    valid_gt = social_gt[finite_mask].float()
    loss = F.binary_cross_entropy_with_logits(
        valid_pred, valid_gt, pos_weight=..., reduction="sum"
    )
    return loss / num_instances.float()               # (3) 유효 개수로만 나눔
```

**차이점 분석:**
| 항목 | 원본 | 구현 코드 |
|------|------|-----------|
| 유효 위치 결정 | `mask` (annotation 유무) | `mask & isfinite(social_pred)` |
| BCE 계산 범위 | 전체 텐서 | valid 위소만 |
| 분모 | `mask.sum() + 1e-6` | `finite_mask.sum().float()` |
| social_gt 타입 | 원본 타입 유지 (int 가능) | `.float()` 명시적 캐스팅 |

**transformer mode에서의 영향:**
- Transformer 모드는 NaN/inf 예측이 발생하지 않으므로 `finite_mask == mask`
- 이 경우 두 구현은 수학적으로 동일
- 단, `+1e-6` 제거로 인해 극단적으로 유효 샘플이 0개일 때 동작 다름 (tensor(0.0) 반환 vs 0/1e-6)
- **사실상 transformer 모드 학습 결과에는 영향 없음**

**graph mode에서의 영향:**
- 초기 학습 시 random-init 된 SocialGraphBlock이 NaN/inf를 생성할 수 있음
- 원본 코드는 NaN이 있으면 loss 전체가 NaN → 학습 파탄
- 구현 코드는 isfinite로 필터링하여 방어적으로 처리
- Graph mode에는 오히려 구현 코드가 더 안전함

---

## [위험도: 중간] 데이터 구성에 영향을 주는 차이

---

### 3. `mtgs/datasets/childplay_temporal.py` — 누락 파일 자동 스킵 (의도된 변경, 유지)

**원본:** 경로 필터링 없음

**구현 코드:**
```python
self.paths = np.array([p for p in self.paths if os.path.exists(os.path.join(root, p))])
```

**영향:** ChildPlay에서 2장의 이미지가 실제로 누락되어 있어 (`js4wxP9HxG0_3635.jpg` 등) 이 필터가 없으면 DataLoader가 FileNotFoundError로 즉시 중단됨. **이 변경은 유지 필요.**

---

## [위험도: 낮음] 기능적으로 동일하지만 구현이 다른 부분

---

### 4. `mtgs/networks/mtgs_net.py` — `speaking_proj` 제거

**원본:**
```python
self.speaking_proj = nn.Linear(1, token_dim)  # __init__에 정의
```
(하지만 `forward()`에서 **사용되지 않음**)

**구현 코드:** 해당 라인 없음

**영향:**
- `speaking_proj`는 원본에서도 `forward()`에서 호출된 적 없는 dead code
- `load_state_dict(strict=False)`이므로 체크포인트에 `speaking_proj` 키가 있어도 그냥 무시됨
- **학습/추론 결과에 영향 없음**

---

### 5. `mtgs/networks/mtgs_net.py` — `interaction_indexes` 계산 방식 변경

**원본:**
```python
self.interaction_indexes = [[0, 2], [3, 5], [6, 8], [9, 11]]  # ViT-B/14 하드코딩
```

**구현 코드:**
```python
chunk = encoder_depth // 4
self.interaction_indexes = [[i * chunk, (i + 1) * chunk - 1] for i in range(4)]
```

**ViT-B/14 (depth=12)에서 결과 비교:**
- chunk = 12 // 4 = 3
- 결과: [[0,2],[3,5],[6,8],[9,11]] — **원본과 동일**
- ViT-L/14 (depth=24)에서는 다름: 원본 [[0,2],[3,5],[6,8],[9,11]] vs 새 코드 [[0,5],[6,11],[12,17],[18,23]]
- **현재 사용 모델(dinov2_vitb14)에서는 영향 없음**

---

### 6. `mtgs/networks/models.py` — transformer mode optimizer param groups

**원본:**
```python
temporal_params = [
    {"params": self.model.gaze_encoder_temporal.parameters(), "lr": lr*3, ...},
    {"params": self.model.people_temporal.parameters(), "lr": lr*3, ...},
    {"params": self.model.decoder_sa.parameters(), "lr": lr*3, ...},
]
other_params = []
for k, v in self.model.named_parameters():
    if ("_temporal" not in k) and ("decoder_sa" not in k):
        other_params.append(v)
# → 4 param groups: [gaze_enc_temporal, people_temporal, decoder_sa, others]
```

**구현 코드 (transformer mode):**
```python
high_lr_prefixes = {"gaze_encoder_temporal", "people_temporal", "decoder_sa"}
high_lr_params = [
    {"params": self.model.gaze_encoder_temporal.parameters(), "lr": lr*3, ...},
    {"params": self.model.people_temporal.parameters(), "lr": lr*3, ...},
    {"params": self.model.decoder_sa.parameters(), "lr": lr*3, ...},
]
other_params = [v for k, v in self.model.named_parameters()
                if not any(k.startswith(p) for p in high_lr_prefixes)]
# → 동일하게 4 param groups
```

**영향:**
- Transformer mode에서 param group 구성, LR 배분, SWA lr 개수 모두 동일
- `"_temporal" not in k` vs `k.startswith("gaze_encoder_temporal") or k.startswith("people_temporal")` → 현재 모델 구조에서 동일한 분류 결과
- **학습 결과에 영향 없음**

---

### 7. `mtgs/networks/models.py` — forward() 반환값 추가 (None, [])

**원본 mtgs_net.py forward:**
```python
return gaze_hm, gaze_vec, inout, lah_logits, laeo_logits, coatt_logits
```

**구현 코드 mtgs_net.py forward:**
```python
return gaze_hm, gaze_vec, inout, lah_logits, laeo_logits, coatt_logits, None, []
# None = null_logits (graph mode 전용), [] = aux_lah_logits (중간 supervision 제거)
```

`models.py`에서도 이에 맞게 `_null, _aux` 언패킹 추가. 일관되게 변경되었으므로 영향 없음.

---

### 8. `mtgs/networks/models.py` — NaN 방어 코드 추가

```python
# 구현 코드에만 추가됨 (train_step, val_step, test_step):
gaze_hm_pred = torch.nan_to_num(gaze_hm_pred, nan=0.0, posinf=0.0, neginf=0.0)
gaze_vec_pred = torch.nan_to_num(gaze_vec_pred, nan=0.0, posinf=0.0, neginf=0.0)
hm_flat = torch.nan_to_num(gaze_hm_pred.reshape(...), nan=0.0)
```

- Transformer mode는 NaN/inf가 발생하지 않으므로 이 guard는 실행되지 않음
- **transformer mode 학습/추론 결과에 영향 없음**

---

### 9. `mtgs/networks/models.py` — test AUC batch_size 버그 수정

**원본:**
```python
self.log("metric/test/auc", test_auc, batch_size=ni, ...)
# ni = (batch["inout"] == 1).sum()  ← 전체 T×N 프레임 기준
```

**구현 코드:**
```python
ni_mid = int((inout_gt == 1).sum().item())  # 중간 프레임만
self.log("metric/test/auc", test_auc, batch_size=ni_mid, ...)
```

**영향:** 테스트 지표인 `metric/test/auc`의 가중 평균 계산이 올바르게 수정됨. 이전에 -3.48 같은 음수 AUC가 나오던 원인이었음. **학습에는 영향 없음, 테스트 결과 보고에만 영향.**

---

### 10. `mtgs/networks/models.py` — test 지표 누적 방식 변경

원본: `test_step_outputs.append(output)` → epoch_end에서 한 번에 처리  
구현 코드: 각 배치를 pickle 파일에 즉시 저장 → epoch_end에서 파일 읽어 처리

**영향:** 메모리 효율 개선 (GPU OOM 방지). 최종 AUC/AP 수치 자체는 동일.

---

## [위험도: 없음] 기능 추가 (graph mode 전용)

다음 변경사항들은 `interaction.type=graph`일 때만 활성화되며, transformer mode 학습/추론에 영향 없음:

| 항목 | 내용 |
|------|------|
| `adaptor_modules.py` — `SocialGraphBlock` 추가 | graph mode 전용 social interaction block |
| `adaptor_modules.py` — `TemporalGraphBlock` 추가 | graph mode 전용 temporal block |
| `adaptor_modules.py` — `GraphGaze` (torch_geometric 기반) **제거** | 기존 미사용 실험 코드 정리 |
| `losses.py` — `compute_null_node_loss()` 추가 | graph mode null node 학습용 (현재 forward에서 `None` 반환이므로 실제 호출 안 됨) |
| `models.py` — graph mode param groups (5개) | graph mode 전용 LR 설정 |
| `train_vsgaze.sh` — `SWA_LR_OVERRIDE` (5개 LR) | graph mode 전용 SWA LR override |
| `config.yaml` — `interaction.graph.*` 섹션 추가 | graph hyperparameter 설정 |

---

## 결론 및 액션 아이템

### VSGaze transformer 재현이 안 되는 원인

| 우선순위 | 파일 | 차이 | 영향 |
|----------|------|------|------|
| **1위** | `scripts/train_vsgaze.sh` | `gradient_clip_val=1.0` 무조건 적용 | **VSGaze transformer 학습 경향 변화** |
| 2위 | `mtgs/datasets/childplay_temporal.py` | 누락 파일 자동 스킵 | 학습 데이터 구성 변경 가능 |
| 3위 | `mtgs/train/losses.py` | `social_loss()` 구현 변경 | transformer 정상 데이터에서는 동일 |

### 즉시 수정해야 할 것

```bash
# train_vsgaze.sh — transformer mode 실행 시:
# 아래 두 줄을 제거하거나 graph mode에만 조건부 적용
train.gradient_clip_val=1.0 \
train.gradient_clip_algorithm=norm \
```

### 선택적 수정 (엄밀한 재현을 위해)

```python
# losses.py social_loss() — 원본으로 되돌리기 (transformer mode 재현성 확보)
def social_loss(social_pred, social_gt, mask, pos_weight=2.0):
    social_gt = social_gt * mask
    num_instances = mask.sum()
    loss = F.binary_cross_entropy_with_logits(
        social_pred, social_gt,
        pos_weight=torch.tensor(pos_weight, device=social_gt.device),
        reduction="none",
    )
    loss = torch.mul(loss, mask).sum() / (num_instances + 1e-6)
    return loss
```

단, graph mode에서 NaN/inf 방어가 필요하다면 분기 처리:
```python
def social_loss(social_pred, social_gt, mask, pos_weight=2.0, safe=False):
    if safe:  # graph mode
        finite_mask = mask & torch.isfinite(social_pred)
        ...
    else:     # transformer mode (원본 동작 유지)
        ...
```
