# Interaction Module 상세 동작 분석

> ⚠️ **DEPRECATED (2026-06-13)**: 이 문서는 `interaction.type` (`transformer`/`graph`/`hypergraph`)
> 스위치와 `SocialGraphBlock`/`UndirectedSocialGraphBlock`/`TemporalGraphBlock`으로 구성됐던
> **삭제된 아키텍처**를 다룬다. 해당 코드는 저장소에 더 이상 존재하지 않으며, `train_vsgaze.sh`를
> 비롯한 현재 파이프라인과 무관하다. 현재 유일한 social-prediction head는 `GazeGraphBlock`이며,
> 그 상세 동작은 [architecture.md](architecture.md)와 [gaze_graph_math.md](gaze_graph_math.md)를
> 참조할 것. 이 문서는 과거 설계의 역사적 참고 자료로만 남겨둔다.

N=3 (person 0, 1, 2), num_layers=2, use_null_node=True, use_gaze_prior=True 가정.

---

## 전체 외부 루프 구조

ViT Adaptor가 4개의 stage로 나뉘므로, 매 stage 이후:

```
stage i (i=0,1,2,3):
  ViT Adaptor[i]               → image_tokens, person_tokens (b*t, N, D)
  sa_in = person_tokens         (directed 블록 직전 trunk 복사)
  # trunk (LAH/LAEO/heatmap)
  SocialGraphBlock[i]          → person_tokens'   (B, N, D) — node features only
  TemporalGraphBlock[i]        → person_tokens''  (t>1 시에만)
  # 평행 SA 분기 (trunk로 안 돌아감)
  UndirectedSocialGraphBlock[i] → sa_tok          (B, N, D)  from sa_in
  sa_temporal_blocks[i]         → sa_tok'         (t>1 시에만)
```

- trunk `person_tokens`는 `gaze_layers[0..3]`에 저장 → heatmap + LAH/LAEO.
- SA 분기 `sa_tok`는 `sa_layers[0..3]`에 저장 → SA(CoAtt) 전용.
- stage 종료 후 pair-wise decoder로 예측: **LAH/LAEO는 trunk, SA는 sa_layers** 사용.

---

## 입력 텐서 (b=1, t=1, N=3, D=768 가정)

```
person_tokens:    (1, 3, 768)  — h0, h1, h2
num_valid_people: (1,)  = [3]
gaze_vecs:        (1, 3, 2)   — 각 사람의 gaze 방향 단위벡터
head_bboxes:      (1, 3, 4)   — 각 사람의 head bbox [x1,y1,x2,y2]
```

---

## Step 1: Edge 인덱스 구성 (캐시, 최초 1회)

N=3에서 방향 있는 모든 pair를 **GT 레이블 순서**로 생성:

```
pairs = [(s,d) for s in range(3) for d in range(3) if s≠d]
      = [(0,1),(0,2),(1,0),(1,2),(2,0),(2,1)]

src_N = [0, 0, 1, 1, 2, 2]   (E=6개)
dst_N = [1, 2, 0, 2, 0, 1]
```

---

## Step 2: Validity Mask

```
num_valid = 3  →  유효 slot = [3-3 .. 3-1] = [0, 1, 2]  → 전부 유효

node_valid:  [True, True, True]          (1, 3)
pair_valid:  3×3 전부 True               (1, 3, 3)
diag_mask:   [[T,F,F],[F,T,F],[F,F,T]]  (1, 3, 3)
```

---

## Step 3: 기하학적 Prior 사전 계산 (루프 밖, 전 iteration 공통)

```python
centers = (head_bboxes[..., :2] + head_bboxes[..., 2:]) / 2   # (1, 3, 2)
```

### LAH prior (attention routing 전용)

각 directed edge (s→d)마다 "s의 gaze가 d 방향을 얼마나 향하나":

```
dir(0→1) = normalize(c1 - c0)
lah_prior[0→1] = cosine(gaze_vec[0], dir(0→1))   ∈ [-1, +1]
lah_prior[0→2] = cosine(gaze_vec[0], dir(0→2))
...  (총 6개)
lah_prior shape: (1, 6)
```

> **주의**: prior는 attention softmax 라우팅에만 사용 (iteration 0). 최종 LAH logit에는 더하지 않음.

---

## Step 4: 내부 반복 (num_layers=2)

### ── Iteration 0 ──

#### 4-0-a. Directed attention score 계산 (Dense N×N)

```python
h_i = h.unsqueeze(2).expand(1, 3, 3, 768)  # h_i[b,i,j] = h[b,i] (소스)
h_j = h.unsqueeze(1).expand(1, 3, 3, 768)  # h_j[b,i,j] = h[b,j] (목적지)
```

**Attention score matrix** — `MLP_dir(cat(h_i, h_j))`:

```
e_dir_mat[0] =
        p0       p1       p2
p0  [  -inf , e(0→1), e(0→2) ]   ← diag = -inf
p1  [ e(1→0),  -inf , e(1→2) ]
p2  [ e(2→0), e(2→1),  -inf  ]
```

**전 iteration에 LAH prior 주입** (각 iter마다 `decay_w[iter_idx]` 비중 다름):

```python
decay_w = softmax(prior_decay_logits)     # (num_layers,)  — 학습됨, 합=1
e_dir_mat += prior_w_attn * decay_w[iter_idx] * lah_prior_mat
# SA prior는 현재 graph 설정에서 directed 블록에 OFF (use_sa_prior=False).
# gaze·gaze SA prior는 UndirectedSocialGraphBlock으로 이동했다.
```

> **주의**: prior는 모든 iteration에 적용되나 `prior_decay_logits`(learnable)의 softmax 가중치로 반복별 비중이 다름. 이전 "iter 0만" 설명은 구버전.

**Dual Null logit** — 분리된 두 MLP로 각각 계산:

```python
e_null_in  = MLP_null_in ([h_i; null_in_node ])   (1, 3)  ← 화면 내 사물 바라봄
e_null_out = MLP_null_out([h_i; null_out_node])   (1, 3)  ← 화면 밖 바라봄
```

---

#### 4-0-b. Softmax: 소스 기준 outgoing edge + dual null

e_dir_mat에 e_null_in, e_null_out을 각각 열로 augment:

```
e_aug[0] =
        p0        p1        p2     null_in  null_out
p0  [  -inf  , e+p(0→1), e+p(0→2), e_ni(0), e_no(0) ]
p1  [ e+p(1→0),  -inf  , e+p(1→2), e_ni(1), e_no(1) ]
p2  [ e+p(2→0), e+p(2→1),  -inf  , e_ni(2), e_no(2) ]

shape: (1, 3, 5)
```

각 행(소스 i)에 대해 softmax → 합이 1:

```
α[0] =
        p0       p1       p2   null_in  null_out
p0  [   0  , α(0→1), α(0→2), α_ni(0), α_no(0) ]   합=1
p1  [ α(1→0),  0   , α(1→2), α_ni(1), α_no(1) ]   합=1
p2  [ α(2→0), α(2→1),  0   , α_ni(2), α_no(2) ]   합=1
```

해석:
- `α(0→1)` = person 0이 person 1을 볼 확률 (outgoing)
- `α_no(0)` = person 0이 화면 밖을 볼 확률 (Null_out — inout supervision)
- `α_ni(0)` = person 0이 화면 내 사물을 볼 확률 (Null_in — no LAH)

---

#### 4-0-c. 메시지 패싱 — outgoing 어텐션 (내가 보는 j들이 기여)

```python
W_msg_h = W_msg(h)   # (1, 3, 768)
msg[b,i] = Σ_j α[b, i, j] * W_msg_h[b, j]
         + α_ni[b,i] * W_msg(null_in_node)
         + α_no[b,i] * W_msg(null_out_node)
```

**person 0의 메시지:**

```
msg[p0] = α(0→1) * W_msg(h1)  +  α(0→2) * W_msg(h2)
         + α_ni(0) * W_msg(null_in_node)
         + α_no(0) * W_msg(null_out_node)
```

- `α(0→1)` 이 크다 = p0이 p1을 보고 있다 → h1이 많이 반영됨
- `α_no(0)` 이 크다 = p0이 화면 밖을 봄 → null_out feature 반영
- `α_ni(0)` 이 크다 = p0이 화면 내 사물을 봄 → null_in feature 반영

---

#### 4-0-d. 노드 업데이트

```python
h_new[i] = update_proj(cat(h[i], msg[i]))   # Linear(1536 → 768)
h[i]     = LayerNorm(h[i] + h_new[i])
h = where(node_valid, h_new, h)              # 패딩 슬롯 보호
```

---

### ── Iteration 1 ──

동일 과정, prior 주입 비중이 `decay_w[1]`로 달라짐 (iter 0는 `decay_w[0]`):

```
e_dir_mat[0] =
        p0       p1       p2
p0  [  -inf , e'(0→1), e'(0→2) ]   ← 갱신된 h로 계산
...
e_dir_mat += prior_w_attn * decay_w[1] * lah_prior_mat   # decay 비중 변화
```

메시지 패싱 → 노드 업데이트.

---

## Step 5: 반환 — 업데이트된 node features + dual-null attention

```python
return h.float(), _alpha_null_in, _alpha_null_out
# (B, N, D), (B, N), (B, N)
```

- `_alpha_null_in`: 마지막 iteration의 `α_ni` — Null_in에 대한 어텐션
- `_alpha_null_out`: 마지막 iteration의 `α_no` — Null_out에 대한 어텐션
- LAH/SA 예측은 downstream의 **공통 pair-wise decoder**가 처리 (transformer 모드와 공유).

---

## Step 6: TemporalGraphBlock (t>1일 때)

```
person_tokens: (b*t, N, D) = (3, 3, 768)
→ reshape: (b*N, t, D) = (3, 3, 768)   [person별로 t 프레임 묶기]
```

각 사람별로 t개 프레임 토큰에 **Multi-Head Self-Attention**:

```
person 0: [h0_t0, h0_t1, h0_t2] → MHA(self-attn over time) → [h0_t0', h0_t1', h0_t2']
person 1: [h1_t0, h1_t1, h1_t2] → MHA → [h1_t0', h1_t1', h1_t2']
person 2: [h2_t0, h2_t1, h2_t2] → MHA → [h2_t0', h2_t1', h2_t2']
```

→ reshape back → `(b*t, N, D) = (3, 3, 768)`

---

## Step 7: Social Gaze 예측 (4 stage 종료 후, 모드별 분기)

```python
# trunk: 4 stage gaze_layers projection 후 concat
proj_tokens = cat([gaze_projs[i](gaze_layers[i]) for i in range(4)])   # (B*T, N, 512)

# SA tokens: graph 모드는 전용 undirected 분기, 그 외는 trunk 재사용
sa_tokens = cat([sa_projs[i](sa_layers[i]) for i in range(4)]) if use_graph else proj_tokens

indices = permutations(range(N), 2)        # N*(N-1)개
```

### 예측 — 전 모드 통합 (decoder_lah / decoder_sa 공유)

```python
# LAH: 비대칭 [h_i ‖ h_j] (방향 i→j), trunk 사용
pair_lah = cat([proj_tokens[:, src], proj_tokens[:, dst]], dim=-1)   # (B*T*P, 1024)
lah   = decoder_lah(pair_lah).view(B*T, num_pairs)

# SA: 대칭 [s_i+s_j ‖ |s_i−s_j|], graph는 SA 분기 / 그 외 trunk
pair_sym = cat([sa_tokens[:, src] + sa_tokens[:, dst],
                (sa_tokens[:, src] - sa_tokens[:, dst]).abs()], dim=-1)
coatt = decoder_sa(pair_sym).view(B*T, num_pairs)

# LAEO ⟺ mutual LAH = logit-space AND (min). rev_idx = (d,s) index per (s,d), n별 캐시
laeo = minimum(lah, lah[:, rev_idx])
```

- 모드 분기는 **SA 입력 토큰(`sa_tokens`)뿐** — graph만 전용 undirected 분기, 나머지는 trunk.
- `decoder_laeo`는 미사용 (dead weight). `lah_null_proj`·`sa_null_w`·`decoder_*_gws` 경로 제거됨.
- `alpha_null_in/out`은 예측이 아니라 `compute_dual_null_loss`(graph 모드)에만 사용.

---

## 4개 outer block의 역할 요약

| Block | ViT Adaptor 단계 | SocialGraphBlock 역할 |
|-------|-----------------|----------------------|
| 0 | 초기 scene↔person cross-attn 후 | 기본적인 pair 관계 학습 (LAH/SA prior 의존 큼) |
| 1 | 중간 scene feature 융합 후 | 1차 refined된 토큰으로 관계 재추정 |
| 2 | 더 deep한 scene feature 후 | 관계가 점점 정교해짐 |
| 3 | 최종 scene feature 후 | 최종 refined person token 출력 |

모든 block의 출력이 `gaze_layers`에 누적되고, **마지막에 한 번** pair-wise decoder로 예측.

---

## 출력 텐서 정리 (SocialGraphBlock 단위)

```
tokens_out:     (1, 3, 768)  — 다음 stage로 전달되는 갱신된 person token
alpha_null_in:  (1, 3)       — Null_in 어텐션 (마지막 iteration)
alpha_null_out: (1, 3)       — Null_out 어텐션 (마지막 iteration)
```

`alpha_null_in_list` / `alpha_null_out_list`에 4 stage 수집 → `compute_dual_null_loss`로 학습.

최종 social gaze 예측 텐서 (forward 전체 기준, Graph 모드):

```
lah:   (B, T, N*(N-1))  — decoder_lah([h_i‖h_j]), trunk(proj_tokens)
laeo:  (B, T, N*(N-1))  — min(LAH_ij, LAH_ji)  (decoder 없음)
coatt: (B, T, N*(N-1))  — decoder_sa([s_i+s_j ‖ |s_i−s_j|]), SA 분기(sa_tokens)
```

`alpha_null_in/out`은 예측에 미사용 — `compute_dual_null_loss` supervision 전용.

---
---

# HypergraphBlock 상세 동작 분석

N=3 (person 0, 1, 2), num_layers=2 가정.  
K = N + 2 = 5 (person hyperedge 3개 + null_in 1개 + null_out 1개).

---

## 전체 외부 루프 구조

graph 모드와 동일하게 4개 stage, 단 SocialGraphBlock → HypergraphBlock으로 대체:

```
stage i (i=0,1,2,3):
  ViT Adaptor[i]        → image_tokens, person_tokens (b*t, N, D)
  HypergraphBlock[i]    → person_tokens'               (B, N, D)
  TemporalGraphBlock[i] → person_tokens''               (t>1 시에만, graph 모드와 동일 모듈)
```

LAH/SA 예측은 종료 후 **공통 pair-wise decoder** 처리 (graph 모드와 동일).

> **interaction.order** 설정에 따라 ViT Adaptor와 HypergraphBlock의 호출 순서가 달라짐.  
> `inject_first` (기본): Adaptor 전체 → HypergraphBlock  
> `extract_first`: Adaptor.extractor → HypergraphBlock → Adaptor.injector+ViT

---

## 입력 텐서 (b=1, t=1, N=3, D=768 가정)

```
person_tokens:    (1, 3, 768)  — h0, h1, h2
num_valid_people: (1,)  = [3]
gaze_vecs:        (1, 3, 2)   — 각 사람의 gaze 방향 단위벡터 (LAH prior용)
head_bboxes:      (1, 3, 4)   — 각 사람의 head bbox [x1,y1,x2,y2] (LAH prior용)
```

---

## Step 1: Valid Mask

```
num_valid = 3  →  유효 slot = [3-3 .. 2] = [0, 1, 2]  → 전부 유효

node_valid:  [True, True, True]           (1, 3)
pair_valid:  3×3 전부 True                (1, 3, 3)
diag_mask:   [[T,F,F],[F,T,F],[F,F,T]]   (1, 3, 3)
```

---

## Step 2: 기하학적 LAH Prior 사전 계산 (루프 밖)

SocialGraphBlock과 동일한 방식:

```python
centers = (head_bboxes[..., :2] + head_bboxes[..., 2:]) / 2   # (1, 3, 2)
dir_ij  = normalize(centers[:, dst] - centers[:, src])         # (1, E, 2)
prior   = (gaze_vecs[:, src] * dir_ij).sum(-1)                 # (1, E) — cosine similarity

lah_prior_mat: (1, 3, 3)  — [src, dst] 위치에 prior 값, 나머지 0
```

iteration 0의 N→E person score에 더해짐.

---

## Step 3: 내부 반복 (num_layers=2)

### ── Iteration 0 ──

#### 3-0-a. Null hyperedge 점수 계산

learnable null 노드 2개 (`null_in_node`, `null_out_node`) 각각 D차원 벡터.

```python
e_null_in [b, i] = MLP_null_in ([h[b,i]; null_in_node ])   # (1, 3, 1)
e_null_out[b, i] = MLP_null_out([h[b,i]; null_out_node])   # (1, 3, 1)
```

---

#### 3-0-b. Person hyperedge 점수 계산 + LAH prior 주입

```python
e_person[b, i, k] = MLP_p([h[b,i]; h[b,k]])   # (1, 3, 3)
```

마스킹:

```
e_person[0] =
        k0(p0)      k1(p1)      k2(p2)
p0  [   -inf    , e_p(0,1), e_p(0,2) ]
p1  [ e_p(1,0),   -inf    , e_p(1,2) ]
p2  [ e_p(2,0), e_p(2,1),   -inf     ]
```

**iter 0에서만 LAH prior 주입** (learnable scalar `prior_w`):

```python
e_person += prior_w * lah_prior_mat
```

---

#### 3-0-c. 전체 Softmax: [null_in | null_out | person_0..2]

```python
e_full  = cat([e_null_in, e_null_out, e_person], dim=-1)  # (1, 3, 5)
attn_full = softmax(e_full, dim=-1)                        # (1, 3, 5)
```

```
e_full[0] =
        null_in     null_out    k0(p0)      k1(p1)      k2(p2)
p0  [ e_ni(0),  e_no(0),     -inf    , e+p(0,1), e+p(0,2) ]
p1  [ e_ni(1),  e_no(1),  e+p(1,0),    -inf    , e+p(1,2) ]
p2  [ e_ni(2),  e_no(2),  e+p(2,0), e+p(2,1),    -inf     ]

attn_full[0] =
        null_in    null_out    k0       k1       k2
p0  [ α_ni(0), α_no(0),    0   , α(0,1), α(0,2) ]  합=1
p1  [ α_ni(1), α_no(1), α(1,0),    0   , α(1,2) ]  합=1
p2  [ α_ni(2), α_no(2), α(2,0), α(2,1),    0    ]  합=1
```

**out-of-frame을 보는 person i**: `α_no(i)` ≈ 1 → person hyperedge에 기여 차단  
**in-frame 사물을 보는 person i**: `α_ni(i)` ≈ 1 → person hyperedge에 기여 차단  
**null option이 항상 finite** → N=1(GazeFollow)에서도 all-inf 문제 없음

```python
attn_person = attn_full[:, :, 2:]   # (1, 3, 3) — person hyperedge 기여분만
```

---

#### 3-0-d. Node → Person Hyperedge 집계 (N→E)

null hyperedge는 집계하지 않음 (흡수만, 되돌려주지 않음):

```python
h_e = einsum("bik,bid->bkd", attn_person, W_e(h))   # (1, 3, 768)
```

```
h_e[k0] = α(1,0)*W_e(h1) + α(2,0)*W_e(h2)   ← "p0를 바라보는 집단" 표현
h_e[k1] = α(0,1)*W_e(h0) + α(2,1)*W_e(h2)   ← "p1을 바라보는 집단" 표현
h_e[k2] = α(0,2)*W_e(h0) + α(1,2)*W_e(h1)   ← "p2를 바라보는 집단" 표현
```

---

#### 3-0-e. E → Node 분배 (비대칭, person hyperedge만)

N→E와 다른 별도 MLP (`MLP_out`)로 분배 가중치 계산 — **self-feedback 차단**:

```python
e_dist[b, i, k] = MLP_out([h[b,i]; h_e[b,k]])        # (1, 3, 3)
e_dist = e_dist.masked_fill(~node_valid[:, None, :], -inf)
attn_dist = softmax(e_dist, dim=-1)                   # (1, 3, 3)

msg = einsum("bik,bkd->bid", attn_dist, W_n(h_e))    # (1, 3, 768)
```

person 0의 메시지:

```
msg[p0] = attn_dist(0,k0)*W_n(h_e[k0])   ← "p0를 바라보는 집단" 정보 (p1,p2가 보면 큼)
         + attn_dist(0,k1)*W_n(h_e[k1])   ← "p1을 바라보는 집단" 정보
         + attn_dist(0,k2)*W_n(h_e[k2])   ← "p2를 바라보는 집단" 정보
```

null hyperedge는 E→N에 참여하지 않음.

**LAEO 포착 메커니즘:**  
p0↔p1 서로 바라볼 경우 → `α(0,1)` 큼 → h_e[k1]에 p0 기여 큼 → msg[p1]에 p0 정보 반영.  
동시에 `α(1,0)` 큼 → h_e[k0]에 p1 기여 큼 → msg[p0]에 p1 정보 반영.  
양방향 모두 반영되므로 LAEO 신호가 양쪽 token에 인코딩됨.

---

#### 3-0-f. 노드 업데이트

```python
gate  = sigmoid(W_gate(h))
delta = update_proj(cat([h, msg], dim=-1))   # Linear(1536 → 768)
h_new = LayerNorm(h + gate * delta)
h     = where(node_valid.unsqueeze(-1), h_new, h)   # 패딩 슬롯 보호
```

---

### ── Iteration 1 ──

업데이트된 `h`로 Step 3-0-a ~ 3-0-f 재실행.  
**LAH prior 주입 없음** (iter 0에서만).

---

## Step 4: 반환

```python
return h.float(), attn_agg.float(), attn_null_in.float()
# (B, N, D), (B, N, N), (B, N)
```

- `attn_agg[i, j]` = 마지막 iteration에서 "person i가 j를 보는 정도" (N→E softmax 중 person 기여분)
- `attn_null_in[i]` = person i의 null_in 어텐션 (`attn_full[:, :, 0]`)
- `attn_agg_hg_list` / `null_in_hg_list`에 4 stage 수집 → social prediction / H-bonus에 활용.

---

## SocialGraphBlock과 HypergraphBlock 비교

| 항목 | SocialGraphBlock (graph 모드) | HypergraphBlock |
|---|---|---|
| Hyperedge 구조 | directed pairwise (N×N) | person hyperedge N개 + null 2개 |
| N→E softmax 선택지 | person_j + null_in + null_out | null_in + null_out + person_k |
| Null node | `null_in_node` + `null_out_node` (별도 MLP 각각) | `null_in_node` + `null_out_node` (동일 구조) |
| E→N routing | outgoing softmax 재사용 (대칭) | 별도 `MLP_out` (비대칭, self-feedback 차단) |
| LAH prior | ✅ 전 iter, `decay_w[iter_idx]` 가중치 | ✅ 전 iter, `decay_w` 가중치 |
| SA prior | ❌ (`use_sa_prior=False` — SA는 undirected 블록 소관) | SA prior 없음 (LAH prior만) |
| 반환값 | `(tokens, α_null_in, α_null_out)` | `(tokens, attn_agg, attn_null_in, attn_null_out)` |
| Social 예측 | LAH=`decoder_lah`, LAEO=`min(LAH)`, SA=`decoder_sa` (전 모드 공통) | 동일 (SA도 trunk 토큰 사용, undirected 분기 없음) |
| Temporal | TemporalGraphBlock (공유) | TemporalGraphBlock (공유) |

---

## 4개 outer block의 역할 요약

| Block | ViT Adaptor 단계 | HypergraphBlock 역할 |
|---|---|---|
| 0 | scene↔person cross-attn 초기 | null 흡수로 out-of-frame 격리, 기본 pair 관계 학습 |
| 1 | 중간 scene feature 융합 | refined token으로 멤버십 재조정 |
| 2 | deeper scene feature | 점차 정교한 집단 구조 형성 |
| 3 | 최종 scene feature | LAH/LAEO 신호 집약된 최종 token |

Social gaze 예측은 graph 모드와 동일하게 4개 stage 종료 후 **공통 pair-wise decoder** 처리.

---
---

# UndirectedSocialGraphBlock 상세 동작 분석 (Graph 모드 SA 전용)

N=3, num_layers=2 가정. **Graph 모드에서만** directed `SocialGraphBlock`과 평행하게 동작.

## 위치와 입출력

```
sa_in = person_tokens (directed 블록 직전 trunk)   # (B, N, D)
sa_tok = UndirectedSocialGraphBlock[i](sa_in, num_valid_people, gaze_vecs)  # (B, N, D)
→ sa_layers[i]에 저장 (trunk로 되먹이지 않음)
```

directed 블록과 **같은 입력**을 보지만 결과는 SA 예측(`decoder_sa`)에만 쓰인다. `head_bboxes`는 안 받음 (LAH 방향 prior 없음).

## SocialGraphBlock과의 핵심 차이

| 항목 | SocialGraphBlock (directed) | UndirectedSocialGraphBlock (SA) |
|---|---|---|
| 대칭성 | 비대칭 (i→j) | **대칭** — edge feature `[h_i+h_j ; |h_i−h_j|]`라 `gate(i,j)==gate(j,i)` |
| 집계 정규화 | softmax over destinations (합=1) + null | **독립 sigmoid 게이트** |
| 메시지 크기 | convex combination | **gated-mean**: `Σ_j gate_ij·W_msg(h_j) / (유효 이웃수)` → N-invariant |
| Null 노드 | null_in/null_out 있음 | **없음** (SA는 "한 명 본다" 배타성 불필요) |
| Prior | LAH 방향 prior | **SA gaze prior** (`gaze_i·gaze_j`, `prior_w_sa` + per-iter decay) |
| 담당 task | LAH/LAEO (+heatmap/inout) | SA (CoAtt) |

## 한 iteration

```python
h_i, h_j = h.unsqueeze(2), h.unsqueeze(1)                 # (B,N,N,D)
e = mlp_edge(cat([h_i+h_j, (h_i-h_j).abs()]))             # (B,N,N) 대칭 logit
e += prior_w_sa * decay_w[it] * sa_prior_mat              # SA prior (옵션)
gate = sigmoid(e) * edge_ok                               # 독립 게이트, 무효/자기 edge=0
msg  = einsum("bij,bjd->bid", gate, W_msg(h)) / deg       # gated-mean (deg=유효 이웃수)
g    = sigmoid(W_gate(h)); delta = update_proj(cat([h, msg]))
h    = where(node_valid, LayerNorm(h + g*delta), h)       # 패딩 슬롯 보호
```

**왜 sigmoid + mean인가**: SA는 "같은 곳을 보냐"라서 비배타적(여러 명과 동시 공유 가능)·대칭. softmax의 "타겟 1명" 경쟁 prior가 부적합하므로 독립 게이트를 쓰고, 군중 크기에 무관하도록 이웃수로 평균낸다.

## 반환

```python
return h.float()   # (B, N, D) — SA-tailored node features
```
null·attention 부산물 없음 (dual-null은 directed 블록 소관).
