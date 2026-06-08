# LLM_TODO.md — Gaze-Graph → Evidence-Augmented LLM Reasoner

> 목적: 현재의 `gaze_graph` mode(Stage 1~3: Visual Evidence Extractor → Directed Gaze
> Graph Builder → Dual-Role Edge Refinement)를 **GMT-style LLM reasoner**(Stage 4~5)로
> 확장한다. 모티브: *Graph-as-Memory Cross-Attention for KGC with LLMs* (arxiv 2026).
> 설계 근거: `2026-06-03.pdf` p.9~18, `graph build.pdf`.

---

## 0. 현재 위치 (이미 구현/학습 중)

```
[1] Visual Evidence Extractor   ─ MTGS 기존 (h_i, g_i, H_i, P_out)         ✅ 동작
[2] Directed Gaze Graph Builder ─ person/region/null nodes, edge init      ✅ 동작
[3] Dual-Role Edge Refinement   ─ row→col→node→refresh ×L, Ê={e^L_{i→t}}   ✅ 동작
       └ Read-out: EdgeDecoder_{lah/sa/null} + BCE                         ✅ 동작 (현재 학습 중)
──────────────────────────────────────────────────────────────────────────
[4] Graph Evidence Tokenizer    ─ Ê → 고정크기 G_LLM tokens                ❌ 미구현
[5] Evidence-Augmented LLM      ─ frozen LLM + cross-attn(G), Yes/No        ❌ 미구현
```

학습 스테이지 매핑:
- **Stage A = Visual-Graph Pretraining** (`L_heatmap + L_inout + L_graph`) ← *지금 진행 중*
- **Stage B = LLM Alignment** (`L_LLM`, Yes/No next-token) ← *이 문서의 주 대상*
- **Stage C = Joint Fine-tuning** (optional, end-to-end) ← 후순위

핵심 원칙: **Stage A의 그래프는 그대로 freeze 가능한 evidence producer**. LLM은 그 위에
얹는 reasoner. 따라서 Stage 4~5는 기존 `gaze_graph` 코드를 **건드리지 않고 추가**하는 방향.

---

## 0.1 선결 작업 — 그래프가 edge state를 내보내게 하기 (BLOCKER)

현재 `GazeGraphBlock.forward`는 logit/mask만 반환한다:
```python
return lah_mat, sa_mat, null_vec, edge_valid     # (B,N,N),(B,N,N),(B,N),(B,N,T)
```
Stage 4는 **refined edge state E `(B,N,T,De)`** 와 **node 정보**가 필요하다. 따라서:

- [ ] `GazeGraphBlock.forward`에 `return_states: bool=False` 인자 추가.
      `True`면 `(lah_mat, sa_mat, null_vec, edge_valid, E, node_src, node_tgt)` 반환.
      (기본 `False`라 Stage A 코드/체크포인트 호환 100% 유지)
- [ ] `E` 는 read-out 직전의 `(B,N,T,De)` 그대로. `edge_valid (B,N,T)`도 같이 넘겨
      tokenizer가 padding/self-edge를 마스킹할 수 있게.
- [ ] target type 구분용 인덱스 규약 재확인: `[0:N]=person, [N:2N]=region, [2N]=null`.

> 이 한 가지만 하면 그래프 쪽 수정은 끝. 나머지는 전부 신규 모듈.

---

## 1. Stage 4 — Graph Evidence Tokenizer

**역할**: 가변 개수의 refined edge set `Ê` → **고정 길이 M개** LLM evidence token `G_LLM`.
(`2026-06-03.pdf` p.15)

### 1.1 Edge token 구성
각 유효 edge `i→t` 에 대해:
```
edge_token(i,t) = MLP([ e^L_{i→t}             # refined relation vector (De)
                        ‖ src_id_emb(i)        # source person 식별 (learnable emb table)
                        ‖ tgt_id_emb(t)        # target 식별
                        ‖ tgt_type_emb(type)   # {person, region, null} 3종
                     ])
```
- id embedding: `nn.Embedding(max_N, d)` (person), region도 동일 인덱스, null은 단일.
  → **순서 불변성**을 위해 id는 "절대 인덱스"가 아니라 의미 없는 슬롯이므로, 가능하면
    id_emb을 빼고 **type_emb + (소스/타깃) node feature**로 대체하는 것도 후보 (결정 D2).
- 입력 set `S_edge = { edge_token(i,t) | edge_valid }`. 크기 = sample마다 다름.

### 1.2 Set-to-Token Attention (압축)
```
Q_g = learnable queries (M, d)                 # nn.Parameter
G   = MHA(query=Q_g, key=S_edge, value=S_edge) # key_padding_mask = ~edge_valid (flatten)
G_LLM = W_proj(G)                              # (M, d_llm) — LLM hidden size에 맞춤
```
- **M (token 수)**: 논문 32. 결정 D3.
- attention layer 수: 1~2 (set-to-seq은 보통 얕게). 결정 D3.
- `W_proj`: graph hidden d → LLM hidden d_llm.

### 1.3 출력
`G_LLM (B, M, d_llm)` — **query-independent, clip-level** evidence. (한 clip당 한 번 계산,
LAH/LAEO/SA 쿼리가 모두 같은 G_LLM을 공유하고 cross-attn으로 다르게 읽음.)

### 1.4 코드 진입점 (신규)
- `mtgs/networks/llm/graph_tokenizer.py` — `GraphEvidenceTokenizer(nn.Module)`.
  입력: `E, edge_valid, (node_src, node_tgt)`. 출력: `G_LLM`.

---

## 2. Stage 5 — Evidence-Augmented LLM Reasoner

**역할**: 자연어 쿼리 + `G_LLM` → Yes/No. (`2026-06-03.pdf` p.16, GMT p.8)

### 2.1 Prompt / Query 형식
세 task를 같은 템플릿군으로:
```
LAH  : "Does <P_i> look at <P_j>?"
LAEO : "Do <P_i> and <P_j> look at each other?"
SA   : "Do <P_i> and <P_j> attend to the same target?"
→ 정답 토큰: "Yes" / "No"
```
- `<P_i>` 는 **special token**. Entity grounding (p.16):
  ```
  Emb(<P_i>) = LLM_token_emb(<P_i>) + W_node · v_i
  ```
  `v_i` = 그래프 person node state (Stage 3 출력). `W_node`: graph d → d_llm.
  → person 식별자를 텍스트가 아니라 **그래프 노드 feature로 grounding**. 결정 D4.

### 2.2 Memory-Augmented Layer (GMT 핵심)
Frozen LLM의 선택된 layer들에 cross-attn 삽입:
```
h = h + FrozenSelfAttn(h)             # frozen
h = h + g_ℓ · CrossAttn(q=h, kv=G_LLM) # ← 학습 대상 (gate g_ℓ는 0 init → 안정적 워밍업)
h = h + FrozenFFN(h)                   # frozen
```
- `g_ℓ`: per-layer 학습 scalar gate, **0으로 init** (초기엔 원본 LLM과 동일 → 붕괴 방지).
- 어느 layer에 삽입? 전체 vs 상위 절반 vs 매 k번째. 결정 D5.
- Base LLM은 **freeze**. 학습 대상 = {cross-attn, gate, W_proj, Q_g, tokenizer, W_node,
  <P_i> embeddings}.

### 2.3 Loss
```
L_LLM = -log P("Yes"|q,G)   (정답 양성)
      = -log P("No" |q,G)   (정답 음성)
```
- autoregressive next-token. 사실상 **answer 토큰 위치의 CE**만 보면 됨 (1 토큰).
- pos/neg 불균형 → class weight 또는 균형 샘플링. 결정 D6.

### 2.4 코드 진입점 (신규)
- `mtgs/networks/llm/reasoner.py` — `EvidenceAugmentedLLM(nn.Module)`.
  HuggingFace `AutoModelForCausalLM` 로드 → frozen → 선택 layer에 `MemoryCrossAttn` 주입
  (forward hook 또는 layer wrapping). `G_LLM`은 forward 인자로 주입.

---

## 3. 데이터 — Yes/No QA 생성

기존 social label을 그대로 재사용 (새 어노테이션 불필요):

| Query | GT 소스 | 양성/음성 |
|-------|---------|-----------|
| LAH  `i→j`     | `lah_labels[i,j]`   | 1=Yes, 0=No, -1=skip |
| LAEO `i,j`     | `laeo_labels` (또는 min(LAH)) | 동일 |
| SA   `i,j`     | `coatt_labels`      | 동일 |

- **샘플 단위**: 한 clip의 유효 pair마다 (task, i, j, label) 튜플 생성 → QA 인스턴스.
- 한 clip = 한 번의 그래프 forward = 한 번의 G_LLM. 그 위에 여러 QA를 배치로 굴림
  (G_LLM 공유 → 효율적).
- **데이터로더**: 기존 vsgaze 배치에서 on-the-fly로 QA 튜플 생성하는 wrapper.
  `mtgs/datasets/gaze_qa.py` (또는 collate 단계에서 생성). 결정 D7.

---

## 4. 학습 스테이지 상세

### Stage A — Visual-Graph Pretraining  *(현재 진행 중)*
- 그대로. 산출물: `gaze_graph`가 잘 학습된 ckpt.

### Stage B — LLM Alignment  *(이 문서 핵심)*
- **freeze**: Visual Evidence Extractor 전체 + (옵션) Dual-Role Refinement까지.
- **train**: tokenizer(Stage4) + cross-attn/gate(Stage5) + grounding(W_node, <P_i>).
- LLM backbone freeze.
- loss: `L_LLM`만 (heatmap/graph loss 끔, 또는 작은 가중치로 유지 — 결정 D8).
- **선행 조건**: Stage A ckpt에서 warm-start. 그래프는 eval 모드로 evidence만 생성.

### Stage C — Joint Fine-tuning  *(optional, 후순위)*
- 그래프 refinement까지 unfreeze해 end-to-end. LLM은 여전히 frozen(+cross-attn 학습).
- 메모리/안정성 이슈 크므로 Stage B 성공 후 검토.

---

## 5. 결정 사항 (Decision Points)

> 구현 전에 정해야 빠르게 진행 가능. 굵게 = 추천 기본값.

- **D1. LLM backbone**: ✅ **확정 = 7B frozen (논문대로, Alpaca-7B 계열)**.
  → 메모리 대책 필수: LLM은 **frozen + bf16/4-bit 로드**(bitsandbytes), gradient 안 흐름.
    학습 대상은 cross-attn/gate/tokenizer 뿐이라 7B여도 학습 param은 작음.
    추론도 frozen이라 KV 캐시 위주. rtx6000(이 환경 96GB) 1장이면 bf16 7B + cross-attn 가능.
  → 7B forward가 병목이므로 **G_LLM은 clip당 1회**, QA는 배치로 묶어 LLM 호출 최소화.
- **D2. Edge token id embedding**: (a) src/tgt id_emb 포함(논문식) vs
  **(b) id 제거하고 type_emb + node feature만** (순서 불변성↑, person 수 가변에 강건).
- **D3. Memory token 수 M / set-to-seq depth**: **M=32, depth=1~2** (논문 따름).
  N이 크면(최대 39) edge 수 폭증 → M 더 키울지 검토.
- **D4. Entity grounding 방식**: **<P_i> special token + W_node·v_i** (p.16).
  대안: 텍스트에 직접 좌표/번호만. grounding이 SA/LAH 성능에 핵심일 가능성.
- **D5. Cross-attn 삽입 위치**: **상위 절반 layer** 기본. 전체 삽입은 비용↑.
- **D6. pos/neg 균형**: BCE class weight vs 균형 샘플링. **clip 내 pos/neg 비율 맞춰 샘플링**.
- **D7. QA 생성 위치**: **collate/loader 단계 on-the-fly** (저장 X).
- **D8. Stage B에서 graph loss 유지 여부**: **L_LLM만** (가장 단순) vs 소가중치 병행.
- **D9. 평가 metric**: ✅ **확정 = 기존 AP 회복/초과가 1차 목표**.
  → `P("Yes")`(정확히는 logit("Yes")−logit("No"))를 pair score로 써서 **기존 LAH/LAEO/SA
    AP 파이프라인 그대로 재사용** → transformer/gaze_graph 표와 직접 비교.
  → 즉 Stage 5는 "자연어 데모"가 아니라 **AP를 끌어올리는 reasoner**로 설계/튜닝.
    자연어 reasoning 데모는 AP 검증 후 확장(후순위).
- **D10. 프레임/temporal**: clip(T frame) → 그래프는 frame별. LLM에 어떤 frame의 G를?
  **중앙 frame 또는 T-pool** (결정 필요). 현재 social label도 clip 단위인지 확인 필요.

---

## 6. 구현 순서 체크리스트

```
[ ] 0.1  GazeGraphBlock.forward(return_states=True) → E, node 노출           (그래프 1곳)
[ ] 1    GraphEvidenceTokenizer 구현 + 단위테스트(가변 N, 마스킹, M 고정)     (신규)
[ ] 3    Yes/No QA 생성기 (loader/collate wrapper)                            (신규)
[ ] 2    EvidenceAugmentedLLM: HF LLM 로드/freeze + MemoryCrossAttn 주입      (신규)
[ ] 2.1  Entity grounding(<P_i> emb + W_node)                                 (신규)
[ ] B    Stage B 학습 스크립트(train_llm_align.sh) + config 블록              (신규)
[ ] D9   평가: P(Yes)→AP, 기존 transformer/gaze_graph 표와 비교
[ ] C    (optional) Joint fine-tuning
```

신규 파일 트리(제안):
```
mtgs/networks/llm/
  ├── graph_tokenizer.py     # Stage 4
  ├── reasoner.py            # Stage 5 (LLM wrap + cross-attn)
  └── memory_attn.py         # MemoryCrossAttn + gate
mtgs/datasets/gaze_qa.py     # Yes/No QA 생성
scripts/train_llm_align.sh   # Stage B
config: interaction.llm.{...} 블록 신설
```

---

## 7. 열린 질문

- ✅ **D1 LLM** — 7B frozen 확정.
- ✅ **목표** — 기존 AP 회복/초과 확정 (P(Yes) score → 기존 AP 파이프라인 재사용).
- [ ] **D10 temporal (착수 시 확인)** — social label이 clip 단위인지 frame 단위인지
      코드에서 확인 필요. 그에 따라 G_LLM을 중앙 frame / T-pool 중 무엇으로 만들지 결정.
      (착수 시 `models.py`의 social loss가 `b*t` 단위인지부터 확인 — 현재 `view(b*t,...)`
      쓰는 걸로 보아 frame 단위일 가능성. 그러면 frame별 G_LLM 또는 T-pool.)

> 핵심 결정 완료. 착수 순서: `0.1 → 1 → 3 → 2 → B` (D10은 3번 QA 생성 시 자연히 결정됨).
