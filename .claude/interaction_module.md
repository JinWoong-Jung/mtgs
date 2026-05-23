# Interaction Module 상세 동작 분석

N=3 (person 0, 1, 2), num_layers=2, use_null_node=True, use_gaze_prior=True 가정.

---

## 전체 외부 루프 구조

ViT Adaptor가 4개의 stage로 나뉘므로, 매 stage 이후:

```
stage i (i=0,1,2,3):
  ViT Adaptor[i]          → image_tokens, person_tokens (b*t, N, D)  ← 언급만
  SocialGraphBlock[i]     → person_tokens', lah_i, sa_i, null_i
  TemporalGraphBlock[i]   → person_tokens''  (t>1 시에만)
```

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
# centers[0] = [c0, c1, c2]  각 사람 head 중심점
```

### LAH prior

각 directed edge (s→d)마다 "s의 gaze가 d 방향을 얼마나 향하나":

```
dir(0→1) = normalize(c1 - c0)
lah_prior[0→1] = cosine(gaze_vec[0], dir(0→1))   ∈ [-1, +1]
lah_prior[0→2] = cosine(gaze_vec[0], dir(0→2))
lah_prior[1→0] = cosine(gaze_vec[1], dir(1→0))
... (총 6개)
lah_prior shape: (1, 6)
```

### SA prior

두 시선 ray가 교차(수렴)하는가:

```
# 직선 교점 파라미터 (t_i, t_j)
det = gaze_s[x] * gaze_d[y] - gaze_s[y] * gaze_d[x]
t_i = ...  ; t_j = ...  (선형계 풀기)

sa_prior[s→d] = (t_i > 0).float() + (t_j > 0).float() - 1.0
              ∈ {-1, 0, +1}
# +1: 두 시선이 서로를 향해 수렴
# -1: 발산
sa_prior shape: (1, 6)
```

---

## Step 4: 내부 반복 (num_layers=2)

### ── Iteration 0 ──

#### 4-0-a. Edge logit 계산 (Dense N×N)

```python
h_i = h.unsqueeze(2).expand(1, 3, 3, 768)  # h_i[b,i,j] = h[b,i] (소스)
h_j = h.unsqueeze(1).expand(1, 3, 3, 768)  # h_j[b,i,j] = h[b,j] (목적지)
```

**LAH logit matrix** — `MLP_dir(cat(h_i, h_j))`:

```
cat shape: (1, 3, 3, 1536) → reshape → (9, 1536) → MLP_dir → (9, 1) → reshape → (1, 3, 3)

e_dir_mat[0] =
        p0       p1       p2
p0  [  -inf , e(0→1), e(0→2) ]   ← diag = -inf
p1  [ e(1→0),  -inf , e(1→2) ]
p2  [ e(2→0), e(2→1),  -inf  ]
```

**iter 0에서만 LAH prior 주입** (learnable scalar `prior_w_attn` 사용):

```
lah_prior_mat[0, src_N, dst_N] = lah_prior[0]
# = [[  0  , p(0→1), p(0→2)],
#    [p(1→0),  0   , p(1→2)],
#    [p(2→0), p(2→1),  0   ]]

e_dir_mat += prior_w_attn * lah_prior_mat   # prior_w_attn: nn.Parameter (init=0.5)
```

**SA logit matrix** — `MLP_sa(h_i + h_j)`:

```
(h_i + h_j) shape: (1, 3, 3, 768) → (9, 768) → MLP_sa → (9, 1) → (1, 3, 3)

e_sa_mat[0] =
        p0        p1        p2
p0  [  s(0,0), s(0,1), s(0,2) ]   ← 대각도 값이 있지만 최종 출력 시 off-diagonal만 사용
p1  [  s(1,0), s(1,1), s(1,2) ]   ← s(i,j) = s(j,i) (대칭)
p2  [  s(2,0), s(2,1), s(2,2) ]
```

**Null logit** — `MLP_null(h_i)`:

```
h.reshape(3, 768) → MLP_null → (3, 1) → reshape → (1, 3)
e_null[0] = [e_null(p0), e_null(p1), e_null(p2)]
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
α[0] =  (row-wise softmax of e_aug)

        p0       p1       p2      null
p0  [   0  , α(0→1), α(0→2), α(0→∅) ]   합=1
p1  [ α(1→0),  0   , α(1→2), α(1→∅) ]   합=1
p2  [ α(2→0), α(2→1),  0   , α(2→∅) ]   합=1
```

해석:
- `α(0→1)` = person 0이 person 1을 볼 확률
- `α(0→∅)` = person 0이 아무도 안 볼 확률

---

#### 4-0-c. 메시지 패싱 — outgoing 어텐션 (내가 보는 j들이 기여)

```python
W_msg_h = W_msg(h)   # (1, 3, 768) — 각 노드의 메시지 벡터
```

`α[:,:,:N]` 을 그대로 사용 (transpose 없음):

```
α[b, i, j] = "i가 j를 볼 확률"  →  "i 기준으로 내가 보는 j들"
```

```
msg[b,i] = Σ_j α[b, i, j] * W_msg_h[b, j]
```

**person 0의 메시지:**

```
msg[p0] = α(0→1) * W_msg(h1)  +  α(0→2) * W_msg(h2)
         + α(0→∅) * W_msg(null_node)
```

- `α(0→1)` 이 크다 = person 0이 1을 보고 있다 → h1이 많이 반영됨
- `α(0→∅)` 이 크다 = 0이 아무도 안 본다 → null feature가 반영됨

**person 1의 메시지:**

```
msg[p1] = α(1→0) * W_msg(h0)  +  α(1→2) * W_msg(h2)
         + α(1→∅) * W_msg(null_node)
```

**person 2의 메시지:**

```
msg[p2] = α(2→0) * W_msg(h0)  +  α(2→1) * W_msg(h1)
         + α(2→∅) * W_msg(null_node)
```

---

#### 4-0-d. 노드 업데이트

```python
h_new[i] = update_proj(cat(h[i], msg[i]))   # Linear(1536 → 768)
h[i]     = LayerNorm(h[i] + h_new[i])        # 잔차 연결 + 정규화
```

h 저장 후 `e_dir_last = e_dir_mat`, `e_sa_last = e_sa_mat`

---

### ── Iteration 1 ──

동일 과정, 단 **LAH prior를 softmax에 주입하지 않음** (attention 안정화된 상태):

```
e_dir_mat[0] =
        p0       p1       p2
p0  [  -inf , e'(0→1), e'(0→2) ]   ← prior 없이 갱신된 h로 계산
p1  [ e'(1→0),  -inf , e'(1→2) ]
p2  [ e'(2→0), e'(2→1),  -inf  ]
```

메시지 패싱 → 노드 업데이트 → `e_dir_last`, `e_sa_last` 갱신.

---

## Step 5: 출력 edge logit 추출 (GT 순서)

```python
lah_logits = e_dir_last[:, src_N, dst_N]   # (1, 6)
# = [e'(0→1), e'(0→2), e'(1→0), e'(1→2), e'(2→0), e'(2→1)]

# 예측용 prior 추가 (learnable scalar prior_w_lah, init=0.5)
lah_logits += prior_w_lah * lah_prior   # (1, 6)
```

```python
sa_logits = e_sa_last[:, src_N, dst_N]   # (1, 6)
# = [s'(0,1), s'(0,2), s'(1,0), s'(1,2), s'(2,0), s'(2,1)]
# s'(i,j) == s'(j,i) (대칭)
sa_logits += prior_w_sa * sa_prior   # learnable scalar prior_w_sa, init=0.5
```

```python
null_logits = MLP_null(h_final)   # (1, 3) — 최종 업데이트된 h에서 계산
```

---

## Step 6: TemporalGraphBlock (t>1일 때)

예를 들어 t=3 (temporal window):

```
person_tokens: (b*t, N, D) = (3, 3, 768)
→ reshape: (b, t, N, D).permute(0,2,1,3).reshape(b*N, t, D) = (3, 3, 768)
  즉 [person0의 3프레임, person1의 3프레임, person2의 3프레임]
```

각 사람별로 t개 프레임 토큰에 **Multi-Head Self-Attention**:

```
person 0: [h0_t0, h0_t1, h0_t2] → MHA(self-attn over time) → [h0_t0', h0_t1', h0_t2']
person 1: [h1_t0, h1_t1, h1_t2] → MHA → [h1_t0', h1_t1', h1_t2']
person 2: [h2_t0, h2_t1, h2_t2] → MHA → [h2_t0', h2_t1', h2_t2']
```

→ reshape back → `(b*t, N, D) = (3, 3, 768)`

---

## 4개 outer block의 역할 요약

| Block | ViT Adaptor 단계 | SocialGraphBlock 역할 |
|-------|-----------------|----------------------|
| 0 | 초기 scene↔person cross-attn 후 | 기본적인 pair 관계 학습 (LAH prior 의존 큼) |
| 1 | 중간 scene feature 융합 후 | 1차 refined된 토큰으로 관계 재추정 |
| 2 | 더 deep한 scene feature 후 | 관계가 점점 정교해짐 |
| 3 | 최종 scene feature 후 | **최종 LAH/SA/null 예측값 생성** |

**Block 0~2의 lah_i** → `aux_lah_logits[0:3]` → 보조 손실 (0.3× 가중치)  
**Block 3의 lah_i** → `lah_from_graph` → 메인 사회적 시선 손실

---

## 출력 텐서 정리

```
tokens_out:  (1, 3, 768)  — 다음 stage로 전달되는 갱신된 person token
lah_logits:  (1, 6)       — LAH(0→1), LAH(0→2), LAH(1→0), LAH(1→2), LAH(2→0), LAH(2→1)
sa_logits:   (1, 6)       — SA(0,1), SA(0,2), SA(1,0), SA(1,2), SA(2,0), SA(2,1)
null_logits: (1, 3)       — Null(0), Null(1), Null(2)
```

LAH는 sigmoid 적용 시 이진 확률; SA도 동일.  
LAEO는 `min(lah_ij, lah_ji)` 로 유도 (별도 head 없음).
