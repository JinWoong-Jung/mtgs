# VLM_TODO.md — Gaze-Graph → Evidence-Augmented VLM Reasoner

> **목적**: 현재의 `gaze_graph` mode(Stage 1~3)를 **GMT-style VLM reasoner**(Stage 4~5)로 확장한다.
> 모티브: *Graph-as-Memory Cross-Attention for KGC with LLMs* (arxiv 2026).
> 설계 근거: `2026-06-03.pdf` p.9~18.

---

## 0. 전체 파이프라인

```
[1] Visual Evidence Extractor   ─ MTGS 기존 (h_i, g_i, H_i, P_out)          ✅ 동작
[2] Directed Gaze Graph Builder ─ person/null_in/null_out, edge init          ✅ 동작
[3] Dual-Role Edge Refinement   ─ row→col→edge→node ×L,  E (B,T,N,Tl,De)    ✅ 동작
       └ Read-out: head_{lah,laeo,sa,null_in,null_out} + BCE               ✅ 동작
──────────────────────────────────────────────────────────────────────────────
[4] Graph Evidence Tokenizer    ─ E[:,t_c] → G_VLM (B,M,d_vlm)              ✅ 구현
[5] Evidence-Augmented VLM      ─ frozen VLM + cross-attn(G) + bbox, Yes/No  ✅ 구현
```

**학습 스테이지:**
- **Stage A** = Visual-Graph Pretraining (`L_heatmap + L_inout + L_graph`) ← 기존 그대로
- **Stage B** = VLM Alignment (`L_VLM`, Yes/No next-token) ← 이 문서의 주 대상
- **Stage C** = Joint Fine-tuning (optional, 후순위)

**핵심 원칙**: Stage A 그래프는 freeze된 evidence producer. VLM은 그 위에 얹는 reasoner.
기존 `gaze_graph` 코드를 **전혀 수정하지 않고** forward hook으로 상태를 추출한다.

---

## 0.1 그래프 상태 노출 방식 — Forward Hook

`GazeGraphBlock` 코드를 수정하지 않고 **외부에서 hook을 등록**한다.

```python
# LLMReasonerModel.__init__ 내부
def _refiner_hook(module, inp, output):
    E, v_src, v_tgt = output          # _UnifiedRefiner 반환값
    self._graph_states["E"] = E.detach()
    self._graph_states["v_src"] = v_src.detach()
    self._graph_states["v_tgt"] = v_tgt.detach()

def _block_hook(module, inp, output):
    self._graph_states["edge_valid"] = output[5].detach()  # 6-tuple의 마지막

self.frozen_mtgs.gaze_graph_block.refiner.register_forward_hook(_refiner_hook)
self.frozen_mtgs.gaze_graph_block.register_forward_hook(_block_hook)
```

**Tensor 규약:**

| Tensor | Shape | 설명 |
|--------|-------|------|
| `E` | `(B, T, N, Tl, De)` | refined edge features. `Tl=N+2`, `De=cfg.interaction.gaze_graph.edge_dim` |
| `v_src` | `(B, T, N, De)` | source person node states |
| `v_tgt` | `(B, T, Tl, De)` | target (person + null) node states |
| `edge_valid` | `(B, N, 2N+2)` | `[0:N]`=p2p, `[N:2N]`=SA proxy, `[2N:2N+2]`=null_in/out |

- `De = cfg.interaction.gaze_graph.edge_dim` (현재 128; config 변경 시 코드 수정 불필요)
- 유효 인물은 **뒤쪽 슬롯** (앞쪽 zero-padding): `valid_start = N - num_valid_people`
- 중앙 frame: `t_c = T // 2` (T=5 → t_c=2)

---

## 1. Stage 4 — Graph Evidence Tokenizer

**파일**: `mtgs/networks/llm/graph_tokenizer.py`

### 1.1 입력 처리

```python
E_c = E[:, t_c]                         # (B, N, Tl, De) — 중앙 frame
# edge_valid (B, N, 2N+2) → Tl=N+2 크기의 마스크로 변환
ev_tl = cat([edge_valid[:,:,:N], edge_valid[:,:,2N:]], dim=2)   # (B, N, Tl)
kv  = E_c.reshape(B, N*Tl, De)          # key-value flatten
kpm = ~ev_tl.reshape(B, N*Tl)           # True = masked out
# 전체 padding 방지: all-masked 시 uniform attention fallback
kpm = kpm & ~kpm.all(dim=1, keepdim=True)
```

> edge_valid의 SA proxy 슬롯 `[N:2N]`은 E_c의 Tl 레이아웃에 없으므로 버린다.
> E_c의 Tl 레이아웃: `[0:N]=person_targets`, `[N]=null_in`, `[N+1]=null_out`.

### 1.2 Set-to-Token Attention (압축)

```
Q_g   = learnable (M, De)               # nn.Parameter, init N(0, 0.02)
for _ in range(depth):
    Q_g = LN(Q_g + MHA(q=Q_g, k=kv, v=kv, key_padding_mask=kpm))
G_VLM = W_proj(Q_g)                     # (B, M, d_vlm)
```

| 하이퍼파라미터 | 값 | config 키 |
|--------------|-----|-----------|
| M (token 수) | 32 | `interaction.llm.memory_tokens_m` |
| depth | 1 | `interaction.llm.tokenizer_depth` |
| num_heads | 8 | 하드코딩 (edge_dim=128, 8 heads → head_dim=16) |

**출력**: `G_VLM (B, M, d_vlm)` — clip당 1회 계산, LAH/LAEO/SA 쿼리 전부 공유.

---

## 2. Stage 5 — Evidence-Augmented VLM Reasoner

**파일**: `mtgs/networks/llm/reasoner.py`

### 2.1 VLM 백본

| 항목 | 값 |
|------|----|
| 모델 | `Qwen/Qwen3-VL-8B-Instruct` |
| 로딩 | `AutoModelForCausalLM(backbone, torch_dtype=bfloat16, trust_remote_code=True)` |
| d_vlm | `model.config.hidden_size` (동적 추출, 하드코딩 불필요) |
| 학습 대상 | `MemoryCrossAttn` × 9, `W_node`, `<P>` embedding, `Q_g`, `W_proj` |
| Freeze | VLM 파라미터 전체 + MTGS 전체 |

### 2.2 Cross-Attention 삽입 — Memory-Augmented Layer

**파일**: `mtgs/networks/llm/memory_attn.py`

Qwen3-VL-8B 텍스트 백본(`Qwen3-8B`)의 구조:
- `num_hidden_layers = 36`, `full_attention_interval = 4`
- Full Attention(GatedAttention) 레이어 인덱스: **[3, 7, 11, 15, 19, 23, 27, 31, 35]** (총 9개)
- Sliding-window attention 레이어는 수정 불필요

각 Full Attention 레이어를 `MemoryAugmentedLayer`로 교체:

```python
h_out = original_layer(h)               # frozen
if G_VLM is not None:
    h_out[0] = h_out[0] + g_ℓ * LN(CrossAttn(q=h_out[0], kv=G_VLM))
```

- `g_ℓ`: per-layer scalar gate, **0으로 init** → 학습 초기에 원본 VLM과 동일 (안정성)
- `G_VLM` 주입: `layer._G_LLM = G_VLM` → forward → `layer._G_LLM = None`

### 2.3 Entity Grounding — 두 레벨 병행

```
텍스트 레벨:  <P> [x1,y1,x2,y2] — head bbox를 프롬프트 텍스트에 삽입 (공간 grounding)
임베딩 레벨: Emb(<P>) += W_node · v    (semantic node grounding)
```

**task별 프롬프트 템플릿:**

| Task | 프롬프트 | subject v | object v |
|------|---------|-----------|----------|
| LAH  | `"Does <P> {src_bbox} look at <P> {dst_bbox}? Answer:"` | `v_src[b,t_c,i]` | `v_tgt[b,t_c,j]` |
| LAEO | `"Do <P> {src_bbox} and <P> {dst_bbox} look at each other? Answer:"` | `v_src[b,t_c,i]` | `v_src[b,t_c,j]` |
| SA   | `"Do <P> {src_bbox} and <P> {dst_bbox} attend to the same target? Answer:"` | `v_src[b,t_c,i]` | `v_src[b,t_c,j]` |

- bbox 포맷: `[{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]` (head_bboxes[b,t_c,idx] 정규화 좌표)
- LAH object에 `v_tgt` 사용 이유: dual-role refiner의 "보여지는 대상" 표현 반영
- `<P>` = vocab에 추가한 단일 special token. 사람 구분은 bbox 텍스트 + node feature 둘 다

### 2.4 Loss

```python
logits = vlm(inputs_embeds=embeds_bf16).logits[0, -1, :]   # 마지막 토큰 위치
log_probs = F.log_softmax(logits.float(), dim=-1)
loss = -log_probs[yes_id if label == 1 else no_id]
L_VLM = mean(loss over all QA pairs in clip)
```

- autoregressive next-token CE, **answer 1 토큰** 위치만 사용
- annotation 없는 pair (`label == -1`) 자동 제외
- Stage B에서는 `L_VLM`만 사용 (graph read-out head loss 없음)

---

## 3. 데이터 — Yes/No QA 생성

**파일**: `mtgs/datasets/gaze_qa.py`

### 3.1 QAPair

```python
@dataclass
class QAPair:
    batch_idx: int
    task: str                                    # "lah" | "laeo" | "sa"
    src_idx: int                                 # looker / first person
    dst_idx: int                                 # target / second person
    label: int                                   # 1=Yes, 0=No
    src_bbox: Tuple[float, float, float, float]  # head_bboxes[b,t_c,src_idx]
    dst_bbox: Tuple[float, float, float, float]  # head_bboxes[b,t_c,dst_idx]
```

### 3.2 GazeQACollator

```
pairs = list(itertools.permutations(range(N_padded), 2))   # mtgs_net.py와 동일 순서
pairs[k] = (src_k, dst_k)
lah_labels[b, t_c, k] = 1  →  src_k가 dst_k를 본다
valid_start = N_padded - num_valid_people[b]               # 유효 인물 우측 정렬
```

- `label == -1` → 제외
- `src_idx < valid_start` or `dst_idx < valid_start` → 패딩 인물 제외
- **1:1 균형 샘플링**: clip당 pos/neg 같은 수, task당 최대 `max_pairs_per_task`(기본 32)

### 3.3 Label 소스

| Task | Batch 키 | 1=Yes 의미 |
|------|---------|-----------|
| LAH  | `lah_labels` (B,T,P) | src가 dst를 본다 |
| LAEO | `laeo_labels` (B,T,P) | 두 사람이 서로 본다 |
| SA   | `coatt_labels` (B,T,P) | 두 사람이 같은 곳을 본다 |

**데이터셋별 annotation 가용성:**

| 데이터셋 | LAH | LAEO | SA |
|---------|-----|------|----|
| VideoAttentionTarget | ✅ | — | — |
| ChildPlay | ✅ | — | — |
| UCO-LAEO | ✅ | ✅ | — |
| VideoCoAtt | — | — | ✅ |

---

## 4. 학습 스테이지

### Stage A — Visual-Graph Pretraining

기존 그대로. 산출물: `gaze_graph` mode로 학습된 ckpt.

### Stage B — VLM Alignment

**파일**: `mtgs/networks/llm/llm_trainer.py`, `mtgs/datasets/llm_datamodule.py`

| 항목 | 설정 |
|------|------|
| Freeze | MTGS + GazeGraphBlock 전체 (eval 모드), VLM backbone |
| Train | `MemoryCrossAttn` × 9, `gate g_ℓ` × 9, `W_node`, `Q_g`, `W_proj`, `<P>` embedding |
| Loss | `L_VLM`만 (read-out head 미사용) |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.01 |
| Precision | bf16-mixed |
| 선행 조건 | Stage A ckpt 로드 (`load_stage_a_weights(ckpt_path)`) |

**Augmentation 정책 — val_transform 강제:**

```python
# LLMDataModule._make_transform() — train/val 동일
Compose([Resize(image_size, head_size), ToTensor(), Normalize(IMG_MEAN, IMG_STD)])
# RandomCropSafeGaze, ColorJitter 없음
```

이유: frozen MTGS는 동일 입력 → 동일 E 보장이 필요. augmentation은 Stage A(시각 인코더 학습)에만 의미가 있음.
효과: `E[:,t_c], v_src[:,t_c], v_tgt[:,t_c], edge_valid`를 결정론적으로 캐싱 가능.

**평가**: `logit("Yes") - logit("No")`를 pair score → 기존 LAH/LAEO/SA AP 파이프라인 재사용.
목표: 기존 `transformer` / `gaze_graph` AP 회복·초과.

### Stage C — Joint Fine-tuning (optional)

GazeGraphBlock refinement까지 unfreeze. VLM은 여전히 frozen. Stage B 성공 후 검토.

---

## 5. 구현된 파일 트리

```
mtgs/networks/llm/
  ├── __init__.py            # 패키지 init (빈 파일)
  ├── graph_tokenizer.py     # GraphEvidenceTokenizer (Stage 4)
  ├── memory_attn.py         # MemoryCrossAttn + MemoryAugmentedLayer
  ├── reasoner.py            # EvidenceAugmentedLLM (Stage 5)
  └── llm_trainer.py         # LLMReasonerModel (pl.LightningModule, Stage B)

mtgs/datasets/
  └── gaze_qa.py             # QAPair + GazeQACollator

  llm_datamodule.py          # LLMDataModule (val_transform 강제)

scripts/
  ├── main_llm.py            # Stage B Hydra 진입점
  └── train_llm_align.sh     # SLURM 스크립트

mtgs/config/config.yaml      # interaction.llm 블록 추가 (additive)
```

---

## 6. Config 블록

`mtgs/config/config.yaml`의 `interaction:` 섹션에 추가된 내용:

```yaml
interaction:
  gaze_graph:
    edge_dim: 128           # De: vlm tokenizer/W_node/W_proj가 동적으로 참조

# ── VLM Reasoner (Stage B: frozen MTGS + trainable cross-attn) ──────────────
vlm:
  backbone: "Qwen/Qwen3-VL-8B-Instruct"    # HF repo 또는 로컬 경로
  load_dtype: "bf16"
  memory_tokens_m: 32                       # G_VLM token 수 M
  tokenizer_depth: 1                        # Set-to-Token MHA depth
  cross_attn_layer_indices: [3,7,11,15,19,23,27,31,35]  # Full Attention 레이어
  gate_init: 0.0                            # g_ℓ 0-init → 안정적 워밍업
  tasks: ["lah", "laeo", "sa"]
  max_pairs_per_task: 32
  lr: 1e-5
  stage_a_ckpt: null                        # 실행 전 필수 설정
```

> `interaction.gaze_graph.edge_dim`(=128)은 VLM tokenizer/W_node가 동적으로 참조 (공유).
> `d_vlm`은 backbone 로드 시 `model.config.hidden_size`에서 자동 추출. 하드코딩 불필요.

---

## 7. Stage B 실행

```bash
# scripts/train_llm_align.sh 의 STAGE_A_CKPT 경로 설정 후:
sbatch scripts/train_llm_align.sh

# 또는 직접 실행:
cd scripts && python main_llm.py \
    vlm.stage_a_ckpt="/path/to/stage_a_gaze_graph.ckpt" \
    experiment.output_folder="../experiments/vlm_align_v1"
```

---

## 8. 검증 체크리스트

```
✅ 기존 코드 수정 없음 (adaptor_modules.py / mtgs_net.py / models.py / vsgaze.py 무수정)
✅ config.yaml은 additive 수정만 (기존 키 변경 없음)
✅ Forward hook으로 E/v_src/v_tgt/edge_valid 추출 (zero change to GazeGraphBlock)
✅ 단위 테스트 13개 통과:
     graph_tokenizer × 3 (shape, all-padding, variable N)
     memory_attn    × 4 (gate=0 identity, shape, no-G passthrough, with-G modifies)
     gaze_qa        × 5 (pairs, skip-1, laeo-skip, balanced, bbox match)
     hook           × 1 (E/v_src/v_tgt shape verification, CPU-only)
✅ Stage A 학습 파이프라인 (gazefollow / vsgaze gaze_graph) 무간섭 확인
```
