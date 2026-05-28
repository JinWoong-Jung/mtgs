# Interaction Module 상세 동작 분석

N=3 (person 0, 1, 2), num_layers=2, use_null_node=True, use_gaze_prior=True 가정.

---

## 전체 외부 루프 구조

ViT Adaptor가 4개의 stage로 나뉘므로, 매 stage 이후:

```
stage i (i=0,1,2,3):
  ViT Adaptor[i]          → image_tokens, person_tokens (b*t, N, D)
  SocialGraphBlock[i]     → person_tokens'          (B, N, D) — node features only
  TemporalGraphBlock[i]   → person_tokens''          (t>1 시에만)
```

모든 4개 stage의 `person_tokens`가 `gaze_layers[0..3]`에 저장되고,
stage 종료 후 **공통 pair-wise decoder**로 LAH/SA 예측.

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

**iter 0에서만 LAH prior 주입** (learnable scalar `prior_w_attn`):

```
e_dir_mat += prior_w_attn * lah_prior_mat
```

**Null logit** — `MLP_null(h_i)`:

```
e_null[0] = [e_null(p0), e_null(p1), e_null(p2)]   (1, 3)
```

---

#### 4-0-b. Softmax: 소스 기준 outgoing edge + null

e_dir_mat에 e_null을 열로 augment:

```
e_aug[0] =
        p0        p1        p2       null
p0  [  -inf  , e+p(0→1), e+p(0→2), e_null(0) ]
p1  [ e+p(1→0),  -inf  , e+p(1→2), e_null(1) ]
p2  [ e+p(2→0), e+p(2→1),  -inf  , e_null(2) ]

shape: (1, 3, 4)
```

각 행(소스 i)에 대해 softmax → 합이 1:

```
α[0] =
        p0       p1       p2      null
p0  [   0  , α(0→1), α(0→2), α(0→∅) ]   합=1
p1  [ α(1→0),  0   , α(1→2), α(1→∅) ]   합=1
p2  [ α(2→0), α(2→1),  0   , α(2→∅) ]   합=1
```

해석:
- `α(0→1)` = person 0이 person 1을 볼 확률 (outgoing)
- `α(0→∅)` = person 0이 아무도 안 볼 확률 (null sink)

---

#### 4-0-c. 메시지 패싱 — outgoing 어텐션 (내가 보는 j들이 기여)

```python
W_msg_h = W_msg(h)   # (1, 3, 768)
msg[b,i] = Σ_j α[b, i, j] * W_msg_h[b, j]
```

**person 0의 메시지:**

```
msg[p0] = α(0→1) * W_msg(h1)  +  α(0→2) * W_msg(h2)
         + α(0→∅) * W_msg(null_node)
```

- `α(0→1)` 이 크다 = p0이 p1을 보고 있다 → h1이 많이 반영됨
- `α(0→∅)` 이 크다 = p0이 아무도 안 본다 → null feature 반영

---

#### 4-0-d. 노드 업데이트

```python
h_new[i] = update_proj(cat(h[i], msg[i]))   # Linear(1536 → 768)
h[i]     = LayerNorm(h[i] + h_new[i])
h = where(node_valid, h_new, h)              # 패딩 슬롯 보호
```

---

### ── Iteration 1 ──

동일 과정, 단 **LAH prior를 softmax에 주입하지 않음** (attention 안정화):

```
e_dir_mat[0] =
        p0       p1       p2
p0  [  -inf , e'(0→1), e'(0→2) ]   ← prior 없이 갱신된 h로 계산
...
```

메시지 패싱 → 노드 업데이트.

---

## Step 5: 반환 — 업데이트된 node features만

```python
return h.float()   # (B, N, D) = (1, 3, 768)
```

SocialGraphBlock은 node feature 업데이트만 담당.  
LAH/SA 예측은 downstream의 **공통 pair-wise decoder**가 처리 (transformer 모드와 동일).

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

## Step 7: Social Gaze 예측 (공통, 4 stage 종료 후)

```python
# 4개 stage person tokens를 projection 후 concat
proj_tokens = cat([gaze_projs[i](gaze_layers[i]) for i in range(4)])
# shape: (B*T, N, 512)

# pair-wise 조합
indices = permutations(range(N), 2)        # N*(N-1)개
opt_1 = proj_tokens[:, indices[0], :]
opt_2 = proj_tokens[:, indices[1], :]
pairs = cat([opt_1, opt_2], dim=-1)         # (B*T*num_pairs, 1024)

lah   = decoder_lah(pairs).view(B*T, num_pairs)   # shared decoder
coatt = decoder_sa(pairs)                          # shared decoder
laeo[pi] = min(lah[pi], lah[corr_idx])             # LAEO 유도
```

---

## 4개 outer block의 역할 요약

| Block | ViT Adaptor 단계 | SocialGraphBlock 역할 |
|-------|-----------------|----------------------|
| 0 | 초기 scene↔person cross-attn 후 | 기본적인 pair 관계 학습 (LAH prior 의존 큼) |
| 1 | 중간 scene feature 융합 후 | 1차 refined된 토큰으로 관계 재추정 |
| 2 | 더 deep한 scene feature 후 | 관계가 점점 정교해짐 |
| 3 | 최종 scene feature 후 | 최종 refined person token 출력 |

모든 block의 출력이 `gaze_layers`에 누적되고, **마지막에 한 번** pair-wise decoder로 예측.

---

## 출력 텐서 정리 (SocialGraphBlock 단위)

```
tokens_out:  (1, 3, 768)  — 다음 stage로 전달되는 갱신된 person token
```

최종 social gaze 예측 텐서 (forward 전체 기준):

```
lah:   (B, T, N*(N-1))  — decoder_lah(pair-wise concat)
laeo:  (B, T, N*(N-1))  — min(lah_ij, lah_ji)
coatt: (B, T, N*(N-1))  — decoder_sa(pair-wise concat)
```
