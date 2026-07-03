# GazeGraphBlock Version History

> 변경 대상 파일: `mtgs/networks/adaptor_modules.py` (별도 명시 없는 한)

---

## 복원 가능 여부 요약

| 버전 | git commit | checkpoint | 코드 복원 |
|------|-----------|------------|----------|
| V6   | `799cc7f` (V6 restored) / `ed8bd41` (V6) | `experiments/V6(SOTA)/train/checkpoints/best.ckpt` | **완벽 복원 가능** |
| V9   | 없음 | `experiments/V9/train/checkpoints/best.ckpt` | 아래 diff로 재현 |
| V10  | 없음 | `experiments/V10/train/checkpoints/best.ckpt` | 아래 diff로 재현 |
| V11  | 없음 | `experiments/V11/train/checkpoints/best.ckpt` | 아래 diff로 재현 |
| V12  | 없음 | `experiments/V12/...` | 아래 diff로 재현 |
| V13  | 없음 | `experiments/V13/...` | 아래 diff로 재현 |
| V14  | 없음 (현재 working tree) | `experiments/V14/...` (학습 예정) | 현재 코드 그대로 (node init 개편 + node 기반 SA) |

---

## V6 — 복원 방법

```bash
git checkout 799cc7f -- mtgs/networks/adaptor_modules.py
# 이후 config/train_vsgaze.sh에서 EXP_NAME, WEIGHTS 설정
```

### V6 핵심 구조

**_UnifiedRefiner (V6 전용 클래스명)**:
- 레이어 가중치 **공유** (ModuleList 아님 — 동일 row/col/refresh 모듈을 num_layers회 반복)
- `self.refresh = MLP(3 * De, De, De)`
- refresh 공식: `E = LN(col + MLP(cat[col, row, col]))` — col-base, sequential, 3×De
- col: row attention 결과 E에 이어서 적용 (not parallel)

**Edge prior**:
- 1채널: `self.linear_edge = nn.Linear(1, De)` (cosine_align만)

**기타**:
- src: heatmap XAttn 있음 (`use_node_xattn`)
- tgt: gated MLP (`tgt_msg_mlp`) — overlap-weighted aggregate + gate
- geom MLP (2-C): **없음**
- temporal edge attn (2-D): **없음**
- SA head: `head_sa(cat[ni_i, ni_j, |ni_i−ni_j|, E[i→j], E[j→i]])` — 5×De

---

## V9 — V6에서 재현

V6 코드(`799cc7f`)를 기반으로 아래 4가지 변경 적용.

### 변경 1: `_UnifiedRefiner` → `_RefinerLayer` + `_UnifiedRefiner`로 분리, ModuleList화 (2-E)

`_RefinerLayer.__init__` (레이어당 독립 모듈):
```python
self.row      = _enc()
self.col      = _enc()
self.temporal = _enc()   # 2-D: temporal edge attention
self.refresh  = MLP(3 * De, De, De)   # V9는 여전히 3×De
self.norm_e   = nn.LayerNorm(De)
```

`_UnifiedRefiner.__init__`에서:
```python
# 기존: self.row = _enc(); self.col = _enc(); self.refresh = MLP(...)
# 변경:
self.layers = nn.ModuleList(
    [_RefinerLayer(edge_dim, heads) for _ in range(num_layers)]
)
```

### 변경 2: Edge prior 2채널로 확장 (2-B)

```python
# 기존: self.linear_edge = nn.Linear(1, De)
# 변경:
self.linear_edge = nn.Linear(2, De)
# prior = [cosine_align, heatmap_overlap]
```

forward에서 feat_all 구성:
```python
feat_p2p = torch.stack([align, overlap], dim=-1)   # (B,T,N,N,2)  ← 기존은 align만
```

### 변경 3: Node geometry MLP 추가 (2-C)

`__init__`:
```python
self.node_geom_mlp = MLP(6, D, D)
nn.init.zeros_(self.node_geom_mlp.fc2.weight)
nn.init.zeros_(self.node_geom_mlp.fc2.bias)
```

`forward` (src_prime 계산 직후):
```python
geom     = torch.cat([centers, wh, gaze_vecs], dim=-1)   # (B,T,N,6)
geom_emb = self.node_geom_mlp(geom)
src_prime = src_prime + geom_emb
# tgt에도: tgt_person_tokens = (...) + geom_emb
```

### 변경 4: Temporal edge attention (2-D, _RefinerLayer.forward 내)

refresh 이후:
```python
if T > 1:
    E_t = E[:, :, :, :, :].permute(0, 2, 3, 1, 4).reshape(B * N * Tl, T, De)
    E_t = self.temporal(E_t)
    E_t = E_t.reshape(B, N, Tl, T, De).permute(0, 3, 1, 2, 4)
    E = self.norm_e(E + E_t) * ev
```

### 변경 5: SA head — pooled gaze-pattern + heatmap grounding (2-A)

`GazeGraphBlock.__init__`:
```python
# 기존 (V6): self.head_sa = _SocialReadoutHead(5 * De)
# V9: per-source pooling projection + overlap scalar
self.sa_pool_proj  = nn.Linear(De, De)        # ψ projection
self.sa_pool_score = nn.Linear(De, 1)          # attention score w_{i→k}
self.sa_overlap_w  = nn.Parameter(torch.zeros(1))   # γ, zero-init
self.head_sa       = _SocialReadoutHead(4 * De)  # cat[E[i], E[j], |diff|, E[i]⊙E[j]]
```

`GazeGraphBlock.forward` (SA readout 섹션):
```python
# gaze-pattern pooling: E[i] = Σ_k w_{i→k}·ψ(E[i→k]),  k ∈ persons + null_in
E_pool = E[:,:,:,:N+1,:]                                       # (B,T,N,N+1,De)
psi    = self.sa_pool_proj(E_pool)                             # (B,T,N,N+1,De)
w      = F.softmax(self.sa_pool_score(psi).squeeze(-1), dim=-1)  # (B,T,N,N+1)
gp     = (w.unsqueeze(-1) * psi).sum(3)                       # (B,T,N,De)

gp_i   = gp.unsqueeze(3).expand(B,T,N,N,De)
gp_j   = gp.unsqueeze(2).expand(B,T,N,N,De)
sa_mat = self.head_sa(
    torch.cat([gp_i, gp_j, (gp_i - gp_j).abs(), gp_i * gp_j], dim=-1)
    .reshape(B*T*N*N, 4*De)
).reshape(B,T,N,N)

# hm_overlap: i·j heatmap 내적 (같은 곳을 응시하는가)
hm_i   = hm_norm.reshape(B,T,N,-1)                            # (B,T,N,Hh*Ww)
hm_ovlp = torch.einsum('btid,btjd->btij', hm_i, hm_i)        # (B,T,N,N)
sa_mat  = sa_mat + torch.sigmoid(self.sa_overlap_w) * hm_ovlp
sa_mat  = (sa_mat + sa_mat.transpose(2,3)) * 0.5
```

공식:
```
E[i]      = Σ_k w_{i→k} · ψ(E[i→k]),   k ∈ {persons, null_in}   (attention pool)
SA[i,j]   = head_sa(cat[E[i], E[j], |E[i]−E[j]|, E[i]⊙E[j]])   # 4·De
           + γ · hm_overlap[i,j]
hm_overlap[i,j] = heatmap[i] · heatmap[j]   (두 사람의 heatmap 내적)
γ = sigmoid(sa_overlap_w),  zero-init
```

### V9 refresh 공식 (V6와 동일)
```
E = LN(col + MLP(cat[col, row, col]))   # 3×De, col-base
```

---

## V10 — V9에서 재현

V9에서 **2가지** 변경.

### 변경 1: SA head 롤백 — V6 방식으로

```python
# 기존 (V9): head_sa = _SocialReadoutHead(4 * De) + sa_pool_proj/score/sa_overlap_w
# 변경 (V10): V6와 동일한 5×De edge 기반으로 롤백
self.head_sa = _SocialReadoutHead(5 * De)
# sa_pool_proj, sa_pool_score, sa_overlap_w 제거

# forward SA readout:
ni     = E[:,:,:,N,:]
ni_i   = ni.unsqueeze(3).expand(B,T,N,N,De)
ni_j   = ni.unsqueeze(2).expand(B,T,N,N,De)
sa_mat = self.head_sa(
    torch.cat([ni_i, ni_j, (ni_i-ni_j).abs(), E_pp, E_pp.transpose(2,3)], dim=-1)
    .reshape(B*T*N*N, 5*De)
).reshape(B,T,N,N)
sa_mat = (sa_mat + sa_mat.transpose(2,3)) * 0.5
```

공식: `SA[i,j] = head_sa(cat[E[i→null_in], E[j→null_in], |diff|, E[i→j], E[j→i]])` — V6와 동일

### 변경 2: refresh 공식 — E_in base로

```python
# _RefinerLayer.__init__:
self.refresh = MLP(3 * De, De, De)   # 크기 동일

# 기존 V9: E = LN(col + MLP(cat[col, row, col]))   # col-base
# 변경 V10:
E = self.norm_e(
    E_in + self.refresh(torch.cat([E_in, row_context, col_context], dim=-1))
) * ev
```

공식: `E = LN(E_in + MLP(cat[E_in, row, col]))` — E_in base, sequential, 3×De

공식: `LN(E_in + MLP(cat[E_in, row, col]))` — E_in base, sequential col, 3×De

---

## V11 — V10에서 재현

V10에서 **refresh 공식 + 크기** 변경 (row 제거).

```python
# _RefinerLayer.__init__:
# 기존: self.refresh = MLP(3 * De, De, De)
self.refresh = MLP(2 * De, De, De)   # 2×De로 축소

# _RefinerLayer.forward:
# row_context는 여전히 계산하지만 refresh에 넣지 않음
E = self.norm_e(
    col_context + self.refresh(torch.cat([col_context, E_in], dim=-1))
) * ev
```

공식: `LN(col + MLP(cat[col, E_in]))` — col-base, 2×De

---

## V12 — V11에서 재현

V11에서 **col을 E_in 기반 parallel로** 변경.

```python
# _RefinerLayer.__init__:
self.refresh = MLP(2 * De, De, De)   # 2×De 유지

# _RefinerLayer.forward:
E_in = E

# ① row: E_in에서
row_context = self.row(
    E_in.reshape(B * T * N, Tl, De), src_key_padding_mask=row_kpm
).reshape(B, T, N, Tl, De) * ev

# ② col: E_in에서 parallel (V11은 row_context 이후 순차 적용)
E_col_in = E_in[:, :, :, :N + 1, :]
E_col_out_N1 = self.col(
    E_col_in.permute(0, 1, 3, 2, 4).reshape(B * T * (N + 1), N, De),
    src_key_padding_mask=col_kpm,
).reshape(B, T, N + 1, N, De).permute(0, 1, 3, 2, 4)
col_context = torch.cat(
    [E_col_out_N1, E_in[:, :, :, N + 1:, :]], dim=3
) * ev

# ③ refresh: E_in residual
E = self.norm_e(
    E_in + self.refresh(torch.cat([row_context, col_context], dim=-1))
) * ev
```

공식: `LN(E_in + MLP(cat[row, col]))` — E_in base, parallel row+col, 2×De

---

## V13 — V12에서 재현

V12에서 3가지 변경 적용.

### 변경 1: tgt node init — XAttn으로 단순화

`GazeGraphBlock.__init__`:
```python
# 제거:
# self.tgt_msg_mlp  = MLP(2 * D, D, D)
# self.tgt_msg_norm = nn.LayerNorm(D)

# 추가:
self.tgt_xattn      = CrossAttention(D, num_heads=heads)
self.tgt_xattn_norm = nn.LayerNorm(D)
```

`GazeGraphBlock.forward` (tgt 구성 섹션):
```python
# 제거 (gated MLP 블록 전체 교체):
kv = src_prime.unsqueeze(2).expand(B, T, N, N, D).reshape(B * T * N, N, D)
tgt_q = person_tokens.reshape(B * T * N, 1, D)

# self-exclusion + padding 마스킹
self_mask = torch.eye(N, dtype=torch.bool, device=device)
self_mask = self_mask.view(1, 1, N, N).expand(B, T, N, N)
inv_mask  = ~valid.unsqueeze(2).expand(B, T, N, N)
kpm       = (self_mask | inv_mask).reshape(B * T * N, N)

tgt_person_tokens = self.tgt_xattn_norm(
    tgt_q + self.tgt_xattn(tgt_q, kv, key_padding_mask=kpm)
).reshape(B, T, N, D) + geom_emb
```

### 변경 2: Edge prior 4채널로 확장

`GazeGraphBlock.__init__`:
```python
# 기존: self.linear_edge = nn.Linear(2, De)
self.linear_edge = nn.Linear(4, De)
```

`GazeGraphBlock.forward` (prior 구성 섹션):
```python
rel_pos  = F.normalize(
    centers.unsqueeze(3) - centers.unsqueeze(2), dim=-1
)                                                                  # (B, T, N, N, 2)
zeros2   = torch.zeros(*null_in_prior.shape, 2, device=device, dtype=dtype)

feat_p2p = torch.cat([align.unsqueeze(-1), overlap.unsqueeze(-1), rel_pos], dim=-1)
                                                                   # (B, T, N, N, 4)
feat_ni  = torch.stack([null_in_prior,  zeros_ch], dim=-1).unsqueeze(3)
feat_ni  = torch.cat([feat_ni, zeros2.unsqueeze(3)], dim=-1)      # (B, T, N, 1, 4)
feat_no  = torch.stack([null_out_prior, zeros_ch], dim=-1).unsqueeze(3)
feat_no  = torch.cat([feat_no, zeros2.unsqueeze(3)], dim=-1)      # (B, T, N, 1, 4)
feat_all = torch.cat([feat_p2p, feat_ni, feat_no], dim=3)         # (B, T, N, Tl, 4)
```

### 변경 3: SA head — mean pool 기반으로 변경

`GazeGraphBlock.__init__`:
```python
# 기존: self.head_sa = _SocialReadoutHead(5 * De)
self.head_sa = _SocialReadoutHead(3 * De)
```

`GazeGraphBlock.forward` (SA readout 섹션):
```python
# person-person + null_in 포함 mean pool (null_out 제외)
E_out  = E[:, :, :, :N + 1, :]                                    # (B, T, N, N+1, De)
ev_out = ev[:, :, :, :N + 1, 0]                                   # (B, T, N, N+1)
r = (E_out * ev_out.unsqueeze(-1)).sum(3) / ev_out.sum(3).clamp(min=1).unsqueeze(-1)
                                                                   # (B, T, N, De)
r_i    = r.unsqueeze(3).expand(B, T, N, N, De)
r_j    = r.unsqueeze(2).expand(B, T, N, N, De)
sa_mat = self.head_sa(
    torch.cat([r_i, r_j, (r_i - r_j).abs()], dim=-1)
    .reshape(B * T * N * N, 3 * De)
).reshape(B, T, N, N)
sa_mat = (sa_mat + sa_mat.transpose(2, 3)) * 0.5
```

---

## V14 — V13에서 재현 (현재 working tree)

> ⚠️ 이전 V14(= V13에서 SA head만 5×De로 롤백)는 **실행취소(폐기)**. V14를 아래 내용으로 재정의함.

**노드 초기화 전면 개편 + SA head를 node 기반으로 분리.** CSGaze 통찰(facial→LAH/LAEO, scene→SA) 반영.

> **차원 설정:** `edge_dim(De)=512`로 상향 실행 (node D=512와 동일). 기존 V6~V13은 De=128/256.
> config: `gaze_graph.edge_dim: 512`. node_proj가 512→512가 되어 노드·엣지 차원이 일치.

핵심 아이디어:
- node = scene(person_token) + face(raw GazeEncoder token 재주입)를 단순 합친 단일 표현. src/tgt 구분 없음.
- 복잡한 src heatmap-XAttn / tgt gated·XAttn 전부 제거 → src=tgt 통합 init.
- LAH/LAEO는 edge 중심 유지, **SA만 node(`v_src`) 기반으로 분리** (성격이 다름).

### 변경 1: 통합 node init (src/tgt 통합, heatmap XAttn 제거)

`GazeGraphBlock.__init__`:
```python
# 제거: hm_proj/hm_pos_emb/hm_pool/src_xattn(+norm), tgt_xattn(+norm),
#       node_src_proj, node_tgt_proj, use_node_xattn 분기
# 추가:
self.face_proj    = nn.Linear(face_dim, D)   # face_dim=768 (raw GazeEncoder), zero-init
nn.init.zeros_(self.face_proj.weight); nn.init.zeros_(self.face_proj.bias)
self.node_in_norm = nn.LayerNorm(D)
self.node_proj    = nn.Linear(D, De)         # src·tgt 공용 단일 projection
```

`GazeGraphBlock.forward`:
```python
geom_emb = self.node_geom_mlp(cat[centers, wh, gaze_vecs])     # (B,T,N,D)
face     = self.face_proj(gaze_feat.detach().to(dtype))        # (B,T,N,D) raw face, grad 차단
node     = self.node_in_norm(person_tokens + face) + geom_emb  # (B,T,N,D)

null_in_t  = null_in_node.view(1,1,1,D).expand(B,T,1,D)
null_out_t = null_out_node.view(1,1,1,D).expand(B,T,1,D)
tgt_tokens = cat[node, null_in_t, null_out_t]                  # (B,T,Tl,D)
v_tgt = self.node_proj(tgt_tokens)                            # (B,T,Tl,De)
v_src = v_tgt[:, :, :N, :]                                    # persons as sources
```
> heatmap은 node에서 빠지고 **edge overlap prior로만** 사용 (gaze_heatmaps 입력 유지).
> plumbing: `mtgs_net.py`가 raw `gaze_tokens`(B,T,N,768)를 `gaze_feat`로 block에 전달, 생성자 `face_dim=token_dim`.

### 변경 2: SA head — node 기반 (edge → node)

`GazeGraphBlock.__init__`:
```python
# 기존: self.head_sa = _SocialReadoutHead(5 * De)   # ni/E_pp edge 기반
self.head_sa = _SocialReadoutHead(2 * De)           # cat(v_src_i, v_src_j)
```

`GazeGraphBlock.forward` (SA readout):
```python
# refined v_src 사용 (refiner의 node-update가 outgoing edge 집계)
v_i = v_src.unsqueeze(3).expand(B, T, N, N, De)
v_j = v_src.unsqueeze(2).expand(B, T, N, N, De)
sa_mat = self.head_sa(
    torch.cat([v_i, v_j], dim=-1).reshape(B * T * N * N, 2 * De)
).reshape(B, T, N, N)
# 대칭화 생략 (asymmetric)
```

> LAH = `head_lah(E[i→j])`, LAEO = `head_laeo(cat[E[i→j], E[j→i]])` (변경 없음).
> rel_pos prior(4채널), temporal/independent layer 등은 V13 그대로 유지.

---

## V15 — V14에서 재현

V14에서 **2가지** 변경. node init(face_proj) 등 나머지는 V14 그대로.

### 변경 1: SA head 롤백 — node 기반 → edge 기반 (V12 방식)

`GazeGraphBlock.__init__`:
```python
# 기존 (V14): self.head_sa = _SocialReadoutHead(2 * De)   # node 기반 cat(v_i, v_j)
self.head_sa = _SocialReadoutHead(5 * De)   # edge 기반
```

`GazeGraphBlock.forward` (SA readout):
```python
ni     = E[:, :, :, N, :]                                          # null_in edge per person
ni_i   = ni.unsqueeze(3).expand(B, T, N, N, De)
ni_j   = ni.unsqueeze(2).expand(B, T, N, N, De)
E_ji   = E_pp.transpose(2, 3)
sa_mat = self.head_sa(
    torch.cat([ni_i, ni_j, (ni_i - ni_j).abs(), E_pp, E_ji], dim=-1)
    .reshape(B * T * N * N, 5 * De)
).reshape(B, T, N, N)
sa_mat = (sa_mat + sa_mat.transpose(2, 3)) * 0.5
```
공식: `SA[i,j] = head_sa(cat[E[i→null_in], E[j→null_in], |diff|, E[i→j], E[j→i]])` — symmetrized

### 변경 2: node-update scoring — query-dependent MLP

`_RefinerLayer.__init__`:
```python
# 기존 (V14): self.pool_out = nn.Linear(De, 1); self.pool_in = nn.Linear(De, 1)
self.pool_out = MLP(2 * De, De, 1)   # MLP(cat[node, edge]) → 1
self.pool_in  = MLP(2 * De, De, 1)
```

`_RefinerLayer.forward` (node update ④):
```python
# out: source node를 query로
v_src_exp  = v_src.unsqueeze(3).expand(B, T, N, Tl, De)
scores_out = self.pool_out(torch.cat([v_src_exp, E], dim=-1)).squeeze(-1)

# in: target node를 query로
E_col       = E[:, :, :, :N + 1, :].permute(0, 1, 3, 2, 4)
v_tgt_exp   = v_tgt[:, :, :N + 1, :].unsqueeze(3).expand(B, T, N + 1, N, De)
scores_in_t = self.pool_in(torch.cat([v_tgt_exp, E_col], dim=-1)).squeeze(-1)
```
스코어 = `MLP(cat[node_state, edge])` → query-independent `Linear(De→1)`에서 노드 상태 의존으로 개선.

---

## V14.5 — V15에서 scoring만 롤백

> V15(SA 롤백 + MLP scoring)에서 **node-update scoring을 V14의 Linear로 되돌린** 버전.
> 즉 **SA head 롤백만 적용**(5×De edge 기반), node-update scoring은 V14와 동일(Linear).
> 목적: "MLP scoring이 실제로 기여하는가"를 분리 검증하는 ablation.

V15 대비 변경 = 위 **변경 2(MLP scoring)를 취소**:
```python
# _RefinerLayer.__init__:
self.pool_out = nn.Linear(De, 1)   # MLP(2*De,De,1) → Linear(De,1)로 롤백
self.pool_in  = nn.Linear(De, 1)

# _RefinerLayer.forward (node update ④):
scores_out  = self.pool_out(E).squeeze(-1)                          # node query 제거
E_col       = E[:, :, :, :N + 1, :].permute(0, 1, 3, 2, 4)
scores_in_t = self.pool_in(E_col).squeeze(-1)
```

| 버전 | SA head | node-update scoring |
|------|---------|---------------------|
| V14   | node 기반 2×De | Linear(De→1) |
| V14.5 | **edge 기반 5×De** | Linear(De→1) |
| V15   | edge 기반 5×De | **MLP(cat[node,edge]→1)** |

---

## 버전별 핵심 변경 비교

| 항목 | V6 | V9 | V10 | V11 | V12 | V13 | V14 |
|------|----|----|-----|-----|-----|-----|-----|
| refresh 입력 | `cat[col,row,col]` | 동일 | `cat[E_in,row,col]` | `cat[col,E_in]` | `cat[row,col]` | 동일 | 동일 |
| refresh 크기 | 3×De | 3×De | 3×De | 2×De | 2×De | 2×De | 2×De |
| residual base | col | col | E_in | col | E_in | E_in | E_in |
| col 입력 | row_ctx 이후 | 동일 | 동일 | 동일 | E_in (parallel) | 동일 | 동일 |
| prior 채널 | 1 | 2 | 2 | 2 | 2 | **4** (+rel_pos) | 4 |
| geom MLP (2-C) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| temporal attn (2-D) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| independent layers (2-E) | ✗ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| src node init | heatmap XAttn | 동일 | 동일 | 동일 | 동일 | 동일 | **person+face, XAttn 제거** |
| tgt init | gated MLP | 동일 | 동일 | 동일 | 동일 | XAttn(src_prime) | **src와 통합(단일 node_proj)** |
| face 재주입 | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (raw gaze_token, zero-init)** |
| SA head | edge 5×De | edge 5×De | edge 5×De | edge 5×De | edge 5×De | edge 3×De (mean) | **node 2×De `cat[v_src_i,v_src_j]`** |

---

## 데이터로더 Shuffle 설정 (`mtgs/datasets/vsgaze.py`)

| split | shuffle | 비고 |
|-------|---------|------|
| train | True | 기존부터 |
| val | **True** (V13~) | 기존 False → 변경. 메트릭은 globally 누적이라 순서 무관 |
| test | True | 기존부터 |
